# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp

from functools import partial

# helpers
from makani.models.common import DropPath, MLP, DecoderWrapper, EncoderWrapper
# Split

# import global convolution and non-linear spectral layers
from makani.models.common import SpectralConv, FactorizedSpectralConv, SpectralAttention

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# wrap fft, to unify interface to spectral transforms
from makani.models.common import RealFFT2, InverseRealFFT2
from makani.mpu.layers import DistributedRealFFT2, DistributedInverseRealFFT2, DistributedMLP, DistributedEncoderDecoder

# more distributed stuff
from makani.utils import comm

# layer normalization
from modulus.distributed.mappings import scatter_to_parallel_region, gather_from_parallel_region
from makani.mpu.layer_norm import DistributedInstanceNorm2d, DistributedLayerNorm

# for annotation of models
import modulus
from modulus.models.meta import ModelMetaData


class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        hidden_size_factor=1,
        factorization=None,
        rank=1.0,
        separable=False,
        complex_activation="real",
        spectral_layers=1,
        bias=False,
        drop_rate=0.0,
        gain=1.0,
        if_encoder=0, # jiaqilong modify
    ):
        super(SpectralFilterLayer, self).__init__()
        # jiaqilong modify
        if if_encoder == 0 :
            in_dim = embed_dim
            out_dim = embed_dim
        else:
            in_dim = embed_dim[0]
            out_dim = embed_dim[1]
        if filter_type == "non-linear":
            self.filter = SpectralAttention(
                forward_transform,
                inverse_transform,
                in_dim,
                out_dim,
                operator_type=operator_type,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=bias,
                gain=gain,
            )

        elif filter_type == "linear" and factorization is None:
            self.filter = SpectralConv(
                forward_transform,
                inverse_transform,
                in_dim,
                out_dim,
                operator_type=operator_type,
                separable=separable,
                bias=bias,
                gain=gain,
            )

        elif filter_type == "linear" and factorization is not None:
            self.filter = FactorizedSpectralConv(
                forward_transform,
                inverse_transform,
                in_dim,
                out_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=bias,
                gain=gain,
            )
        # end
        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.Identity, nn.Identity),
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,
        use_mlp=False,
        comm_feature_inp_name=None,
        comm_feature_hidden_name=None,
        complex_activation="real",
        spectral_layers=1,
        bias=False,
        final_activation=False,
        checkpointing=0,
        if_encoder=0 # jiaqlong modify
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        # determine some shapes
        if comm.get_size("spatial") > 1:
            self.input_shape_loc = (forward_transform.lat_shapes[comm.get_rank("h")],
                                    forward_transform.lon_shapes[comm.get_rank("w")])
            self.output_shape_loc = (inverse_transform.lat_shapes[comm.get_rank("h")],
                                     inverse_transform.lon_shapes[comm.get_rank("w")])
        else:
            self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
            self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear":
            # jiaqilong modify
            if if_encoder == 0:
                self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
                gain_factor /= 2.0
                nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / embed_dim))
            else:
                self.inner_skip = nn.Conv2d(embed_dim[0], embed_dim[1], 1, 1, bias=False)
                gain_factor /= 2.0
                nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / embed_dim[1]))
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()
            gain_factor /= 2.0
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            hidden_size_factor=mlp_ratio,
            factorization=factorization,
            rank=rank,
            separable=separable,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            bias=bias,
            drop_rate=path_drop_rate,
            gain=gain_factor,
            if_encoder=if_encoder
        )

        self.act_layer0 = act_layer()

        # norm layer
        self.norm1 = norm_layer[1]()

        if final_activation and act_layer != nn.Identity:
            gain_factor = 2.0
        else:
            gain_factor = 1.0

        if outer_skip == "linear":
            # self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
            if if_encoder == 0:
                self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
                gain_factor /= 2.0
                torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / embed_dim))
            else:
                self.outer_skip = nn.Conv2d(embed_dim[0], embed_dim[1], 1, 1, bias=False)
                gain_factor /= 2.0
                torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / embed_dim[1]))
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()
            gain_factor /= 2.0
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            if if_encoder == 0:
                mlp_hidden_dim = int(embed_dim * mlp_ratio)
                self.mlp = MLPH(
                    in_features=embed_dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop_rate=mlp_drop_rate,
                    drop_type="features",
                    comm_inp_name=comm_feature_inp_name,
                    comm_hidden_name=comm_feature_hidden_name,
                    checkpointing=checkpointing,
                    gain=gain_factor,
                )
            else:
                mlp_hidden_dim = int(embed_dim[1] * mlp_ratio)
                self.mlp = MLPH(
                    in_features=embed_dim[1],
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer,
                    drop_rate=mlp_drop_rate,
                    drop_type="features",
                    comm_inp_name=comm_feature_inp_name,
                    comm_hidden_name=comm_feature_hidden_name,
                    checkpointing=checkpointing,
                    gain=gain_factor,
                )
            

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        if final_activation:
            self.act_layer1 = act_layer()

    def forward(self, x):
        """
        Updated FNO block
        """

        x, residual = self.filter(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer0"):
            x = self.act_layer0(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        if hasattr(self, "act_layer1"):
            x = self.act_layer1(x)

        return x


class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SFNO implementation as in Bonev et al.; Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    """

    def __init__(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        filter_type="linear",
        operator_type="dhconv",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        scale_factor=8,
        inp_chans=2,
        out_chans=2,
        embed_dim=32,
        num_layers=4,
        repeat_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        encoder_ratio=1,
        decoder_ratio=1,
        activation_function="gelu",
        encoder_layers=1,
        pos_embed="none",
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="instance_norm",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        big_skip=True,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_activation="real",
        spectral_layers=3,
        bias=False,
        checkpointing=0,
        **kwargs,
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.repeat_layers = repeat_layers
        self.big_skip = big_skip
        self.checkpointing = checkpointing

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # initialize spectral transforms
        self._init_spectral_transforms(spectral_transform, model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # set up encoder
        if comm.get_size("matmul") > 1:
            self.encoder = DistributedEncoderDecoder(
                num_layers=encoder_layers,
                input_dim=self.inp_chans,
                output_dim=self.embed_dim,
                hidden_dim=int(encoder_ratio * self.embed_dim),
                act_layer=activation_function,
                input_format="nchw",
                comm_inp_name="fin",
                comm_out_name="fout",
            )
            fblock_mlp_inp_name = self.encoder.comm_out_name
            fblock_mlp_hidden_name = "fout" if (self.encoder.comm_out_name == "fin") else "fin"
        else:
            #=self.encoder = EncoderDecoder(
            #    num_layers=encoder_layers,
            #    input_dim=self.inp_chans,
            #    output_dim=self.embed_dim,
            #    hidden_dim=int(encoder_ratio * self.embed_dim),
            #    act_layer=activation_function,
            #    input_format="nchw",
            #)
            
            self.encoder = EncoderWrapper(inp_chans, encoder_layers, self.embed_dim, activation_function)
            fblock_mlp_inp_name = "fin"
            fblock_mlp_hidden_name = "fout"

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # pick norm layer
        if normalization_layer == "layer_norm":
            norm_layer_inp = partial(DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6)
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_inp = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_inp = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "none":
            norm_layer_out = norm_layer_mid = norm_layer_inp = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            first_layer = i == 0
            last_layer = i == num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "linear"

            if first_layer:
                norm_layer = (norm_layer_inp, norm_layer_mid)
            elif last_layer:
                norm_layer = (norm_layer_mid, norm_layer_out)
            else:
                norm_layer = (norm_layer_mid, norm_layer_mid)

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
                comm_feature_inp_name=fblock_mlp_inp_name,
                comm_feature_hidden_name=fblock_mlp_hidden_name,
                rank=rank,
                factorization=factorization,
                separable=separable,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                bias=bias,
                checkpointing=checkpointing,
            )

            self.blocks.append(block)

        # decoder takes the output of FNO blocks and the residual from the big skip connection
        if comm.get_size("matmul") > 1:
            comm_inp_name = fblock_mlp_inp_name
            comm_out_name = fblock_mlp_hidden_name
            self.decoder = DistributedEncoderDecoder(
                num_layers=encoder_layers,
                input_dim=embed_dim,
                output_dim=self.out_chans,
                hidden_dim=int(decoder_ratio * embed_dim),
                act_layer=activation_function,
                gain=0.5 if self.big_skip else 1.0,
                comm_inp_name=comm_inp_name,
                comm_out_name=comm_out_name,
                input_format="nchw",
            )
            self.gather_shapes = compute_split_shapes(self.out_chans,
                                                      comm.get_size(self.decoder.comm_out_name))

        else:
            self.decoder = DecoderWrapper(
                num_layers=encoder_layers,
                input_dim=embed_dim,
               # output_dim=self.out_chans,
                hidden_dim=int(decoder_ratio * embed_dim),
                act_layer=activation_function,
              #  gain=0.5 if self.big_skip else 1.0,
              #  input_format="nchw",
            )

        # output transform
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.inp_chans, self.out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.inp_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # learned position embedding
        if pos_embed == "direct":
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1]))
            # information about how tensors are shared / sharded across ranks
            self.pos_embed.is_shared_mp = []  # no reduction required since pos_embed is already serial
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]
            self.pos_embed.type = "direct"
            with torch.no_grad():
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed == "frequency":
            if comm.get_size("spatial") > 1:
                lmax_loc = self.itrans_up.l_shapes[comm.get_rank("h")]
                mmax_loc = self.itrans_up.m_shapes[comm.get_rank("w")]
            else:
                lmax_loc = self.itrans_up.lmax
                mmax_loc = self.itrans_up.mmax

            rcoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc), diagonal=0))
            ccoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc - 1), diagonal=-1))
            with torch.no_grad():
                nn.init.trunc_normal_(rcoeffs, std=0.02)
                nn.init.trunc_normal_(ccoeffs, std=0.02)
            self.pos_embed = nn.ParameterList([rcoeffs, ccoeffs])
            self.pos_embed.type = "frequency"
            self.pos_embed.is_shared_mp = []
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]

        elif pos_embed == "none" or pos_embed == "None" or pos_embed == None:
            pass
        else:
            raise ValueError("Unknown position embedding type")

    @torch.jit.ignore
    def _init_spectral_transforms(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)

        # prepare the spectral transforms
        if spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # parallelism
            if comm.get_size("spatial") > 1:
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
                sht_handle = thd.DistributedRealSHT
                isht_handle = thd.DistributedInverseRealSHT

            # set up
            self.trans_down = sht_handle(*self.inp_shape, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type).float()
            self.itrans_up = isht_handle(*self.out_shape, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type).float()
            self.trans = sht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()
            self.itrans = isht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()

        elif spectral_transform == "fft":
            fft_handle = RealFFT2
            ifft_handle = InverseRealFFT2

            if comm.get_size("spatial") > 1:
                h_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                w_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(h_group, w_group)
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(*self.inp_shape, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up = ifft_handle(*self.out_shape, lmax=modes_lat, mmax=modes_lon).float()
            self.trans = fft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans = ifft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if comm.get_size("spatial") > 1:
            self.inp_shape_loc = (self.trans_down.lat_shapes[comm.get_rank("h")],
                                  self.trans_down.lon_shapes[comm.get_rank("w")])
            self.out_shape_loc = (self.itrans_up.lat_shapes[comm.get_rank("h")],
                                  self.itrans_up.lon_shapes[comm.get_rank("w")])
            self.h_loc = self.itrans.lat_shapes[comm.get_rank("h")]
            self.w_loc = self.itrans.lon_shapes[comm.get_rank("w")]
        else:
            self.inp_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.out_shape_loc = (self.itrans_up.nlat, self.itrans_up.nlon)
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):
        for r in range(self.repeat_layers):
            for blk in self.blocks:
                if self.checkpointing >= 3:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)

        return x

    def forward(self, x):
        # save big skip
        if self.big_skip:
            # if output shape differs, use the spectral transforms to change resolution
            if self.out_shape != self.inp_shape:
                xtype = x.dtype
                # only take the predicted channels as residual
                residual = x.to(torch.float32)
                with amp.autocast(enabled=False):
                    residual = self.trans_down(residual)
                    residual = residual.contiguous()
                    residual = self.itrans_up(residual)
                    residual = residual.to(dtype=xtype)
            else:
                # only take the predicted channels
                residual = x

        if comm.get_size("fin") > 1:
            x = scatter_to_parallel_region(x, 1, "fin")

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):
            if self.pos_embed.type == "frequency":
                pos_embed = torch.stack([self.pos_embed[0], nn.functional.pad(self.pos_embed[1], (1, 0), "constant", 0)], dim=-1)
                with amp.autocast(enabled=False):
                    pos_embed = self.itrans_up(torch.view_as_complex(pos_embed))
            else:
                pos_embed = self.pos_embed

            # add pos embed
            x = x + pos_embed

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)

        if hasattr(self.decoder, "comm_out_name") and (comm.get_size(self.decoder.comm_out_name) > 1):
            x = gather_from_parallel_region(x, 1, self.gather_shapes, self.decoder.comm_out_name)

        if self.big_skip:
            x = x + self.residual_transform(residual)
        
        return x

# this part exposes the model to modulus by constructing modulus Modules
@dataclass
class SphericalFourierNeuralOperatorNetMetaData(ModelMetaData):
    name: str = "SFNO"

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True

SFNO = modulus.Module.from_torch(
    SphericalFourierNeuralOperatorNet,
    SphericalFourierNeuralOperatorNetMetaData()
)

class FourierNeuralOperatorNet(SphericalFourierNeuralOperatorNet):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, spectral_transform="fft", **kwargs)

@dataclass
class FourierNeuralOperatorNetMetaData(ModelMetaData):
    name: str = "FNO"

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True

FNO = modulus.Module.from_torch(
    FourierNeuralOperatorNet,
    FourierNeuralOperatorNetMetaData()
)

# class FourierNeuralOperatorNet(SphericalFourierNeuralOperatorNet):
#     def __init__(self, *args, **kwargs):
#         return super().__init__(*args, spectral_transform="fft", **kwargs)

# jiaqilong modify
class Encoder_sfno(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 act,
                 trans_down,
                trans,
                itrans,
                filter_type,
                operator_type,
                mlp_ratio,
                mlp_drop_rate,
                path_drop_rate,
                # act_layer,
                norm_layer,
                inner_skip,
                outer_skip,
                use_mlp,
                comm_feature_inp_name,
                comm_feature_hidden_name,
                rank,
                factorization,
                separable,
                complex_activation,
                spectral_layers,
                bias,
                checkpointing,
                if_cube = 1):
        super(Encoder_sfno, self).__init__()

        encoder_modules = []
        current_dim = input_dim 
        for i in range(num_layers):
            block = FourierNeuralOperatorBlock(
            trans_down,
            itrans,
            (input_dim, hidden_dim),
            filter_type=filter_type,
            operator_type=operator_type,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            path_drop_rate=path_drop_rate,
            act_layer=act,
            norm_layer=norm_layer,
            inner_skip=inner_skip,
            outer_skip=outer_skip,
            use_mlp=use_mlp,
            comm_feature_inp_name=comm_feature_inp_name,
            comm_feature_hidden_name=comm_feature_hidden_name,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            bias=bias,
            checkpointing=checkpointing,
            if_encoder=1,)
            encoder_modules.append(block)
            # nn.init.kaiming_normal_(encoder_modules[-1].weight, mode="fan_in", nonlinearity="relu")
            encoder_modules.append(act())
            current_dim = hidden_dim
            
        encoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False))
        nn.init.kaiming_normal_(encoder_modules[-1].weight, mode="fan_in", nonlinearity="linear")
        self.fwd = nn.Sequential(*encoder_modules)
        self.if_cube = if_cube
        if if_cube == 1:
            self.cube_embedding = nn.Conv1d(2, 4, kernel_size=1, stride=1, padding='valid')
            nn.init.kaiming_normal_(self.cube_embedding.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        # print(x.shape)
        if self.if_cube == 1:
            N, _, D, H, W = x.shape
            x = x.view(N, 2, -1)
            x = self.cube_embedding(x) 
            x = x.view(N, D*4, H, W)  # magic number: 4
            cube = x
            return self.fwd(x), cube
        else:
            N, D, H, W = x.shape
            return self.fwd(x)


class Split(nn.Module):
    def __init__(self, layers0, layers1, layers2, split_size, dim):
        super(Split, self).__init__()
        self.split_size = split_size
        self.dim = dim
        self.layers0 = nn.ModuleList(layers0)
        self.layers1 = nn.ModuleList(layers1)
        self.layers2 = nn.ModuleList(layers2)

    def forward(self, x):
        y = []
        res0 = []
        res1 = []
        res2 = []
        d = []
        # y = torch.cat([layer(input) for (layer, input) in zip(self.layers, torch.split(x, self.split_size, dim=self.dim))], dim = 1)
        for (layer0, layer1, layer2, input) in zip(self.layers0, self.layers1, self.layers2, torch.split(x, self.split_size, dim=self.dim)):
            x0, cube_result = checkpoint(layer0, input, use_reentrant=False)
            res0.append(cube_result)
            # print('cube_result', cube_result.shape)
            res1.append(x0)
            x1 = checkpoint(layer1, x0, use_reentrant=False)
            res2.append(x1)
            x2 = checkpoint(layer2, x1, use_reentrant=False)
            y.append(x2)
        res0 = torch.cat(res0, dim=1)
        res1 = torch.cat(res1, dim=1)
        res2 = torch.cat(res2, dim=1)
        y = torch.cat(y, dim=1)
        # print('res0', res0.shape)
        # print('res1', res1.shape)
        return y, res0, res1, res2

class EncoderWrapper_sfno(nn.Module):
    def __init__(self, inp_chans, num_layers, hidden_dim, act_layer,
                    trans_down,
                    itrans_up,
                    trans,
                    itrans,
                    filter_type,
                    operator_type,
                    mlp_ratio,
                    mlp_drop_rate,
                    path_drop_rate,
                    # act_layer,
                    norm_layer,
                    inner_skip,
                    outer_skip,
                    use_mlp,
                    comm_feature_inp_name,
                    comm_feature_hidden_name,
                    rank,
                    factorization,
                    separable,
                    complex_activation,
                    spectral_layers,
                    bias,
                    checkpointing):
        super(EncoderWrapper_sfno, self).__init__()
        split_size = [8,13,13,13,13,13]
        input_dims = [i * 4 for i in split_size]  # magic number: 4
        # output_dims = [60,60,60,60,60,60]
        mid_dims = [64, 64, 64, 64, 64, 64]
        output_dims = [96,96,96,96,96,96]
        # output_dims = [100,100,100,100,100,100]
        embed_dim = 576
        norm_layer_quarter = partial(nn.InstanceNorm2d, num_features=embed_dim // 4, eps=1e-6, affine=True, track_running_stats=False)
        norm_layer_half = partial(nn.InstanceNorm2d, num_features=embed_dim // 2, eps=1e-6, affine=True, track_running_stats=False)
        norm_layer_all = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        encoders = [Encoder_sfno(num_layers, i, o, hidden_dim // 4, act_layer,
                                trans_down = trans_down[3], # (origin_h, origin_w) - > origin
                                trans = trans,
                                itrans = itrans_up[3], # origin origin
                                filter_type=filter_type,
                                operator_type=operator_type,
                                mlp_ratio=mlp_ratio,
                                mlp_drop_rate=mlp_drop_rate,
                                path_drop_rate=path_drop_rate,
                                # act_layer=act_layer,
                                norm_layer=(norm_layer_quarter, norm_layer_quarter),
                                inner_skip=inner_skip,
                                outer_skip=outer_skip,
                                use_mlp=use_mlp,
                                comm_feature_inp_name=comm_feature_inp_name,
                                comm_feature_hidden_name=comm_feature_hidden_name,
                                rank=rank,
                                factorization=factorization,
                                separable=separable,
                                complex_activation=complex_activation,
                                spectral_layers=spectral_layers,
                                bias=bias,
                                checkpointing=checkpointing, if_cube=1) for i, o in zip(input_dims, mid_dims)]
        # first downsample
        encoders_0 = [Encoder_sfno(num_layers, i, o, hidden_dim // 2, act_layer,
                                trans_down = trans_down[0], # (origin_h, origin_w) - > mid
                                trans = trans,
                                itrans = itrans_up[2], # mid -> (mid_h, mid_w)
                                filter_type=filter_type,
                                operator_type=operator_type,
                                mlp_ratio=mlp_ratio,
                                mlp_drop_rate=mlp_drop_rate,
                                path_drop_rate=path_drop_rate,
                                # act_layer=act_layer,
                                norm_layer=(norm_layer_half, norm_layer_half),
                                inner_skip=inner_skip,
                                outer_skip=outer_skip,
                                use_mlp=use_mlp,
                                comm_feature_inp_name=comm_feature_inp_name,
                                comm_feature_hidden_name=comm_feature_hidden_name,
                                rank=rank,
                                factorization=factorization,
                                separable=separable,
                                complex_activation=complex_activation,
                                spectral_layers=spectral_layers,
                                bias=bias,
                                checkpointing=checkpointing, if_cube=0) for i, o in zip(mid_dims, output_dims)]
        encoders_1 = [Encoder_sfno(num_layers, i, o, hidden_dim, act_layer,
                                trans_down = trans_down[1], #(mid_h, mid_w) - > model
                                trans = trans,
                                itrans = itrans, #model -> (h, w)
                                filter_type=filter_type,
                                operator_type=operator_type,
                                mlp_ratio=mlp_ratio,
                                mlp_drop_rate=mlp_drop_rate,
                                path_drop_rate=path_drop_rate,
                                # act_layer=act_layer,
                                norm_layer=(norm_layer_all, norm_layer_all),
                                inner_skip=inner_skip,
                                outer_skip=outer_skip,
                                use_mlp=use_mlp,
                                comm_feature_inp_name=comm_feature_inp_name,
                                comm_feature_hidden_name=comm_feature_hidden_name,
                                rank=rank,
                                factorization=factorization,
                                separable=separable,
                                complex_activation=complex_activation,
                                spectral_layers=spectral_layers,
                                bias=bias,
                                checkpointing=checkpointing, if_cube=0) for i, o in zip(output_dims, output_dims)]
        self.split = Split(encoders, encoders_0, encoders_1, split_size, dim=2)
        # self.additional_encoder = MLP(in_features=inp_chans-73*2, hidden_features=hidden_dim, out_features=hidden_dim-sum(output_dims))
        # self.additional_encoder0 = Encoder_sfno(num_layers, inp_chans-73*2, hidden_dim, hidden_dim, act_layer,
        #                                         trans_down = trans_down[0], # (origin_h, origin_w) - > mid
        #                                         trans = trans,
        #                                         itrans = itrans_up[2], # mid -> (mid_h, mid_w)
        #                                         filter_type=filter_type,
        #                                         operator_type=operator_type,
        #                                         mlp_ratio=mlp_ratio,
        #                                         mlp_drop_rate=mlp_drop_rate,
        #                                         path_drop_rate=path_drop_rate,
        #                                         # act_layer=act_layer,
        #                                         norm_layer=norm_layer,
        #                                         inner_skip=inner_skip,
        #                                         outer_skip=outer_skip,
        #                                         use_mlp=use_mlp,
        #                                         comm_feature_inp_name=comm_feature_inp_name,
        #                                         comm_feature_hidden_name=comm_feature_hidden_name,
        #                                         rank=rank,
        #                                         factorization=factorization,
        #                                         separable=separable,
        #                                         complex_activation=complex_activation,
        #                                         spectral_layers=spectral_layers,
        #                                         bias=bias,
        #                                         checkpointing=checkpointing,
        #                                         if_cube=0) 
        # self.additional_encoder1 = Encoder_sfno(num_layers, hidden_dim, hidden_dim-sum(output_dims), hidden_dim, act_layer,
        #                                         trans_down = trans_down[1], #(mid_h, mid_w) - > model
        #                                         trans = trans,
        #                                         itrans = itrans, #model -> (h, w)
        #                                         filter_type=filter_type,
        #                                         operator_type=operator_type,
        #                                         mlp_ratio=mlp_ratio,
        #                                         mlp_drop_rate=mlp_drop_rate,
        #                                         path_drop_rate=path_drop_rate,
        #                                         # act_layer=act_layer,
        #                                         norm_layer=norm_layer,
        #                                         inner_skip=inner_skip,
        #                                         outer_skip=outer_skip,
        #                                         use_mlp=use_mlp,
        #                                         comm_feature_inp_name=comm_feature_inp_name,
        #                                         comm_feature_hidden_name=comm_feature_hidden_name,
        #                                         rank=rank,
        #                                         factorization=factorization,
        #                                         separable=separable,
        #                                         complex_activation=complex_activation,
        #                                         spectral_layers=spectral_layers,
        #                                         bias=bias,
        #                                         checkpointing=checkpointing,
        #                                         if_cube=0) 

    def forward(self, x):
        # reshape input
        # print('x.shape', x.shape)
        x, additional_features = x[:, :73 *2, :, :], x[:, 73 *2:, :, :]
        x = x.view(x.shape[0], 2, 73, x.shape[2], x.shape[3])
        # y = self.split(x)
        # res_origin: input shape, res0: input shape, res1: input shape // scale factor
        y, res_cube, res0, res1 = checkpoint(self.split, x, use_reentrant=False)
        # a = checkpoint(self.additional_encoder0, additional_features, use_reentrant=False)
        # a = checkpoint(self.additional_encoder1, a, use_reentrant=False)
        # print('y', y.shape)
        # print('a', a.shape)
        # y = torch.cat(y, dim=1)
        # return torch.cat([y, a], dim=1), res0, res1
        return y, res_cube, res0, res1

class Gather(nn.Module):
    def __init__(self, layers0, layers1, layers2, hidden_dim):
        super(Gather, self).__init__()
        self.layers0 = nn.ModuleList([layers0])
        self.layers1 = nn.ModuleList([layers1])
        self.layers2 = nn.ModuleList(layers2)
        self.res_mid_conv = nn.Conv2d(hidden_dim // 2 + 576, hidden_dim // 2, 1, bias=False)
        self.act = nn.GELU()
        self.res_origin_conv = nn.Conv2d(hidden_dim // 4 + 384, hidden_dim // 4, 1, bias=False)
        nn.init.kaiming_normal_(self.res_mid_conv.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.res_origin_conv.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        ipt = x[0]
        mid_res = x[1]
        origin_res = x[2]
        # y = self.layers0[0](ipt)
        y = checkpoint(self.layers0[0], ipt, use_reentrant=False)

        y = torch.cat([y, mid_res], dim=1)
        y = checkpoint(self.res_mid_conv, y, use_reentrant=False)
        y = self.act(y)
        y = checkpoint(self.layers1[0], y, use_reentrant=False)

        y = torch.cat([y, origin_res], dim=1)
        y = checkpoint(self.res_origin_conv, y, use_reentrant=False)
        y = self.act(y)
        # y = self.res_mid_conv(y)
        result = []
        for layer2 in self.layers2:
            x0 = checkpoint(layer2, y, use_reentrant=False)
            result.append(x0)
        re = torch.cat(result, dim=1)
        return re

class Decoder_sfno(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 act,
                 trans_down,
                trans,
                itrans_up,
                filter_type,
                operator_type,
                mlp_ratio,
                mlp_drop_rate,
                path_drop_rate,
                # act_layer,
                norm_layer,
                inner_skip,
                outer_skip,
                use_mlp,
                comm_feature_inp_name,
                comm_feature_hidden_name,
                rank,
                factorization,
                separable,
                complex_activation,
                spectral_layers,
                bias,
                checkpointing,):
        super(Decoder_sfno, self).__init__()

        decoder_modules = []
        current_dim = hidden_dim
        for i in range(num_layers):
            block = FourierNeuralOperatorBlock(
            trans,
            itrans_up,
            (hidden_dim, hidden_dim),
            filter_type=filter_type,
            operator_type=operator_type,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            path_drop_rate=path_drop_rate,
            act_layer=act,
            norm_layer=norm_layer,
            inner_skip=inner_skip,
            outer_skip=outer_skip,
            use_mlp=use_mlp,
            comm_feature_inp_name=comm_feature_inp_name,
            comm_feature_hidden_name=comm_feature_hidden_name,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            bias=bias,
            checkpointing=checkpointing,
            if_encoder=1,)
            decoder_modules.append(block)
            # nn.init.kaiming_normal_(encoder_modules[-1].weight, mode="fan_in", nonlinearity="relu")
            decoder_modules.append(act())
            current_dim = hidden_dim
            
        decoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False))
        nn.init.kaiming_normal_(decoder_modules[-1].weight, mode="fan_in", nonlinearity="linear")
        self.fwd = nn.Sequential(*decoder_modules)
        # self.if_cube = if_cube
        # if if_cube == 1:
        #     self.cube_embedding = nn.Conv1d(2, 4, kernel_size=1, stride=1, padding='valid')
        #     nn.init.kaiming_normal_(self.cube_embedding.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        # print(x.shape)
        # return self.fwd(x)
        return checkpoint(self.fwd, x, use_reentrant=False)


class DecoderWrapper_sfno(nn.Module):
    def __init__(self, num_layers, hidden_dim, act_layer,
                    trans_down,
                    itrans_up,
                    trans,
                    itrans,
                    filter_type,
                    operator_type,
                    mlp_ratio,
                    mlp_drop_rate,
                    path_drop_rate,
                    # act_layer,
                    norm_layer,
                    inner_skip,
                    outer_skip,
                    use_mlp,
                    comm_feature_inp_name,
                    comm_feature_hidden_name,
                    rank,
                    factorization,
                    separable,
                    complex_activation,
                    spectral_layers,
                    bias,
                    checkpointing):
        super(DecoderWrapper_sfno, self).__init__()
        input_dim = []
        # output_dims = [60,60,60,60,60,60]
        output_dims = [8,13,13,13,13,13]
        # output_dims = [100,100,100,100,100,100]
        # first downsample

        embed_dim = 576
        norm_layer_quarter = partial(nn.InstanceNorm2d, num_features=embed_dim // 4, eps=1e-6, affine=True, track_running_stats=False)
        norm_layer_half = partial(nn.InstanceNorm2d, num_features=embed_dim // 2, eps=1e-6, affine=True, track_running_stats=False)
        norm_layer_all = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)

        decoders_0 = Decoder_sfno(num_layers, hidden_dim, hidden_dim // 2, hidden_dim, act_layer,
                                trans_down = trans_down[0], # (origin_h, origin_w)
                                trans = trans,
                                itrans_up = itrans_up[0], # (mid_h, mid_w)
                                filter_type=filter_type,
                                operator_type=operator_type,
                                mlp_ratio=mlp_ratio,
                                mlp_drop_rate=mlp_drop_rate,
                                path_drop_rate=path_drop_rate,
                                # act_layer=act_layer,
                                norm_layer=(norm_layer_all, norm_layer_all),
                                inner_skip=inner_skip,
                                outer_skip=outer_skip,
                                use_mlp=use_mlp,
                                comm_feature_inp_name=comm_feature_inp_name,
                                comm_feature_hidden_name=comm_feature_hidden_name,
                                rank=rank,
                                factorization=factorization,
                                separable=separable,
                                complex_activation=complex_activation,
                                spectral_layers=spectral_layers,
                                bias=bias,
                                checkpointing=checkpointing)
        decoders_1 = Decoder_sfno(num_layers, hidden_dim, hidden_dim // 4, hidden_dim // 2, act_layer,
                                trans_down = trans_down[0], # (origin_h, origin_w)
                                trans = trans_down[2],
                                itrans_up = itrans_up[1], # (mid_h, mid_w)
                                filter_type=filter_type,
                                operator_type=operator_type,
                                mlp_ratio=mlp_ratio,
                                mlp_drop_rate=mlp_drop_rate,
                                path_drop_rate=path_drop_rate,
                                # act_layer=act_layer,
                                norm_layer=(norm_layer_half, norm_layer_half),
                                inner_skip=inner_skip,
                                outer_skip=outer_skip,
                                use_mlp=use_mlp,
                                comm_feature_inp_name=comm_feature_inp_name,
                                comm_feature_hidden_name=comm_feature_hidden_name,
                                rank=rank,
                                factorization=factorization,
                                separable=separable,
                                complex_activation=complex_activation,
                                spectral_layers=spectral_layers,
                                bias=bias,
                                checkpointing=checkpointing) 
        decoders_2 = [Decoder_sfno(num_layers, hidden_dim, o, hidden_dim // 4, act_layer,
                                trans_down = trans_down[1], #(h, w)
                                trans = trans_down[3],
                                itrans_up = itrans_up[3], #(h, w)
                                filter_type=filter_type,
                                operator_type=operator_type,
                                mlp_ratio=mlp_ratio,
                                mlp_drop_rate=mlp_drop_rate,
                                path_drop_rate=path_drop_rate,
                                # act_layer=act_layer,
                                norm_layer=(norm_layer_quarter, norm_layer_quarter),
                                inner_skip=inner_skip,
                                outer_skip=outer_skip,
                                use_mlp=use_mlp,
                                comm_feature_inp_name=comm_feature_inp_name,
                                comm_feature_hidden_name=comm_feature_hidden_name,
                                rank=rank,
                                factorization=factorization,
                                separable=separable,
                                complex_activation=complex_activation,
                                spectral_layers=spectral_layers,
                                bias=bias,
                                checkpointing=checkpointing) for o in output_dims]
        self.gather = Gather(decoders_0, decoders_1, decoders_2, hidden_dim)

    def forward(self, x, mid_res, origin_res):
        return checkpoint(self.gather, [x, mid_res, origin_res] , use_reentrant=False)


class SphericalFourierNeuralOperatorNetSfnoEnc(nn.Module):
    """
    SFNO implementation as in Bonev et al.; 
    with cube embed encoder;
    with sfno encoder & decoder
    Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    """

    def __init__(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        filter_type="linear",
        operator_type="dhconv",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        scale_factor=8,
        inp_chans=2,
        out_chans=2,
        embed_dim=32,
        num_layers=4,
        repeat_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        encoder_ratio=1,
        decoder_ratio=1,
        activation_function="gelu",
        encoder_layers=1,
        pos_embed="none",
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="instance_norm",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        big_skip=True,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_activation="real",
        spectral_layers=3,
        bias=False,
        checkpointing=0,
        **kwargs,
    ):
        super(SphericalFourierNeuralOperatorNetSfnoEnc, self).__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.repeat_layers = repeat_layers
        self.big_skip = big_skip
        self.checkpointing = checkpointing

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor // scale_factor)

        # h & w after first downsample
        self.h_mid = int(self.inp_shape[0] // scale_factor)
        self.w_mid = int(self.inp_shape[1] // scale_factor)
        # initialize spectral transforms
        self._init_spectral_transforms(spectral_transform, model_grid_type, sht_grid_type, hard_thresholding_fraction, max_modes)

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # set up encoder
        if comm.get_size("matmul") > 1:
            self.encoder = DistributedEncoderDecoder(
                num_layers=encoder_layers,
                input_dim=self.inp_chans,
                output_dim=self.embed_dim,
                hidden_dim=int(encoder_ratio * self.embed_dim),
                act_layer=activation_function,
                input_format="nchw",
                comm_inp_name="fin",
                comm_out_name="fout",
            )
            fblock_mlp_inp_name = self.encoder.comm_out_name
            fblock_mlp_hidden_name = "fout" if (self.encoder.comm_out_name == "fin") else "fin"
        else:
            inner_skip = "none"
            outer_skip = "linear"
            
            fblock_mlp_inp_name = "fin"
            fblock_mlp_hidden_name = "fout"

            norm_layer_all = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer = (norm_layer_all, norm_layer_all)
            self.encoder = EncoderWrapper_sfno(inp_chans, encoder_layers, self.embed_dim, activation_function,
                                                trans_down = self.trans_down,
                                                itrans_up = self.itrans_up,
                                                trans = self.trans,
                                                itrans = self.itrans,
                                                filter_type=filter_type,
                                                operator_type=operator_type,
                                                mlp_ratio=mlp_ratio,
                                                mlp_drop_rate=mlp_drop_rate,
                                                path_drop_rate=0,
                                                # act_layer=activation_function,
                                                norm_layer=norm_layer,
                                                inner_skip=inner_skip,
                                                outer_skip=outer_skip,
                                                use_mlp=use_mlp,
                                                comm_feature_inp_name=fblock_mlp_inp_name,
                                                comm_feature_hidden_name=fblock_mlp_hidden_name,
                                                rank=rank,
                                                factorization=factorization,
                                                separable=separable,
                                                complex_activation=complex_activation,
                                                spectral_layers=spectral_layers,
                                                bias=bias,
                                                checkpointing=checkpointing,)

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # pick norm layer
        if normalization_layer == "layer_norm":
            norm_layer_inp = partial(DistributedLayerNorm, normalized_shape=(embed_dim), elementwise_affine=True, eps=1e-6)
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_inp = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_inp = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "none":
            norm_layer_out = norm_layer_mid = norm_layer_inp = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for ii in range(num_layers - 2): # jiaqilong modify num_layer -> num_layer - 1
            i = ii + 2
            first_layer = i == 0
            last_layer = i == num_layers - 1
            forward_transform = self.trans
            # if i == num_layers - 1:
            #     inverse_transform = self.itrans_up[1]
            #     forward_transform = self.trans_down[1]
            # elif i == num_layers - 2:
            #     inverse_transform = self.itrans_up[0]
            # else:
            inverse_transform = self.itrans
            # forward_transform = self.trans_down if first_layer else self.trans
            # inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "linear"

            if first_layer:
                norm_layer = (norm_layer_inp, norm_layer_mid)
            elif last_layer:
                norm_layer = (norm_layer_mid, norm_layer_out)
            else:
                norm_layer = (norm_layer_mid, norm_layer_mid)

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
                comm_feature_inp_name=fblock_mlp_inp_name,
                comm_feature_hidden_name=fblock_mlp_hidden_name,
                rank=rank,
                factorization=factorization,
                separable=separable,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                bias=bias,
                checkpointing=checkpointing,
            )

            self.blocks.append(block)

        # decoder takes the output of FNO blocks and the residual from the big skip connection
        if comm.get_size("matmul") > 1:
            comm_inp_name = fblock_mlp_inp_name
            comm_out_name = fblock_mlp_hidden_name
            self.decoder = DistributedEncoderDecoder(
                num_layers=encoder_layers,
                input_dim=embed_dim,
                output_dim=self.out_chans,
                hidden_dim=int(decoder_ratio * embed_dim),
                act_layer=activation_function,
                gain=0.5 if self.big_skip else 1.0,
                comm_inp_name=comm_inp_name,
                comm_out_name=comm_out_name,
                input_format="nchw",
            )
            self.gather_shapes = compute_split_shapes(self.out_chans,
                                                      comm.get_size(self.decoder.comm_out_name))

        else:
            self.decoder = DecoderWrapper_sfno(1, self.embed_dim, activation_function,
                                                trans_down = self.trans_down,
                                                itrans_up = self.itrans_up,
                                                trans = self.trans,
                                                itrans = self.itrans,
                                                filter_type=filter_type,
                                                operator_type=operator_type,
                                                mlp_ratio=mlp_ratio,
                                                mlp_drop_rate=mlp_drop_rate,
                                                path_drop_rate=0,
                                                # act_layer=activation_function,
                                                norm_layer=norm_layer,
                                                inner_skip=inner_skip,
                                                outer_skip=outer_skip,
                                                use_mlp=use_mlp,
                                                comm_feature_inp_name=fblock_mlp_inp_name,
                                                comm_feature_hidden_name=fblock_mlp_hidden_name,
                                                rank=rank,
                                                factorization=factorization,
                                                separable=separable,
                                                complex_activation=complex_activation,
                                                spectral_layers=spectral_layers,
                                                bias=bias,
                                                checkpointing=checkpointing,)
        self.cube_conv = nn.Conv2d(self.out_chans * 5, self.out_chans, 1, bias=False)
        # output transform
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.inp_chans, self.out_chans, 1, bias=False)
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.inp_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # learned position embedding
        if pos_embed == "direct":
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1]))
            # information about how tensors are shared / sharded across ranks
            self.pos_embed.is_shared_mp = []  # no reduction required since pos_embed is already serial
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]
            self.pos_embed.type = "direct"
            with torch.no_grad():
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed == "frequency":
            if comm.get_size("spatial") > 1:
                lmax_loc = self.itrans_up.l_shapes[comm.get_rank("h")]
                mmax_loc = self.itrans_up.m_shapes[comm.get_rank("w")]
            else:
                lmax_loc = self.itrans_up.lmax
                mmax_loc = self.itrans_up.mmax

            rcoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc), diagonal=0))
            ccoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc - 1), diagonal=-1))
            with torch.no_grad():
                nn.init.trunc_normal_(rcoeffs, std=0.02)
                nn.init.trunc_normal_(ccoeffs, std=0.02)
            self.pos_embed = nn.ParameterList([rcoeffs, ccoeffs])
            self.pos_embed.type = "frequency"
            self.pos_embed.is_shared_mp = []
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]

        elif pos_embed == "none" or pos_embed == "None" or pos_embed == None:
            pass
        else:
            raise ValueError("Unknown position embedding type")

    @torch.jit.ignore
    def _init_spectral_transforms(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)
            modes_lat_input = int(self.inp_shape[0] * hard_thresholding_fraction)
            modes_lon_input = int((self.inp_shape[1] // 2 + 1) * hard_thresholding_fraction)
            modes_lat_mid = int(self.h_mid * hard_thresholding_fraction)
            modes_lon_mid = int((self.w_mid // 2 + 1) * hard_thresholding_fraction)

        # prepare the spectral transforms
        if spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # parallelism
            if comm.get_size("spatial") > 1:
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
                sht_handle = thd.DistributedRealSHT
                isht_handle = thd.DistributedInverseRealSHT

            # set up
            # only modify unparallelism

            self.trans_down = [sht_handle(*self.inp_shape, lmax=modes_lat_mid, mmax=modes_lon_mid, grid=model_grid_type).float(), 
                               sht_handle(self.h_mid, self.w_mid, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type).float(),
                               sht_handle(self.h_mid, self.w_mid, lmax=modes_lat_mid, mmax=modes_lon_mid, grid=model_grid_type).float(),
                               sht_handle(*self.inp_shape, lmax=modes_lat_input, mmax=modes_lon_input, grid=model_grid_type).float()] 
            # self.itrans_up = [isht_handle(self.h_mid, self.w_mid, lmax=modes_lat_mid, mmax=modes_lon_mid, grid=model_grid_type).float(), isht_handle(*self.out_shape, lmax=modes_lat_input, mmax=modes_lon_input, grid=model_grid_type).float()]
            self.itrans_up = [isht_handle(self.h_mid, self.w_mid, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type).float(),
                              isht_handle(*self.out_shape, lmax=modes_lat_mid, mmax=modes_lon_mid, grid=model_grid_type).float(),
                              isht_handle(self.h_mid, self.w_mid, lmax=modes_lat_mid, mmax=modes_lon_mid, grid=model_grid_type).float(),
                              isht_handle(*self.out_shape, lmax=modes_lat_input, mmax=modes_lon_input, grid=model_grid_type).float()]
            self.trans = sht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()
            self.itrans = isht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()

        elif spectral_transform == "fft":
            fft_handle = RealFFT2
            ifft_handle = InverseRealFFT2

            if comm.get_size("spatial") > 1:
                h_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                w_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(h_group, w_group)
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(*self.inp_shape, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up = ifft_handle(*self.out_shape, lmax=modes_lat, mmax=modes_lon).float()
            self.trans = fft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans = ifft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if comm.get_size("spatial") > 1:
            self.inp_shape_loc = (self.trans_down.lat_shapes[comm.get_rank("h")],
                                  self.trans_down.lon_shapes[comm.get_rank("w")])
            self.out_shape_loc = (self.itrans_up.lat_shapes[comm.get_rank("h")],
                                  self.itrans_up.lon_shapes[comm.get_rank("w")])
            self.h_loc = self.itrans.lat_shapes[comm.get_rank("h")]
            self.w_loc = self.itrans.lon_shapes[comm.get_rank("w")]
        else:
            self.inp_shape_loc = (self.trans_down[0].nlat, self.trans_down[0].nlon)
            self.out_shape_loc = (self.itrans_up[1].nlat, self.itrans_up[1].nlon)
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):
        for r in range(self.repeat_layers):
            for blk in self.blocks:
                if self.checkpointing >= 3:
                    # print(x.shape)
                    # print(x.shape, res_mid.shape)
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)

        return x

    def forward(self, x):
        # save big skip
        if self.big_skip:
            # if output shape differs, use the spectral transforms to change resolution
            if self.out_shape != self.inp_shape:
                xtype = x.dtype
                # only take the predicted channels as residual
                residual = x.to(torch.float32)
                with amp.autocast(enabled=False):
                    residual = self.trans_down(residual)
                    residual = residual.contiguous()
                    residual = self.itrans_up(residual)
                    residual = residual.to(dtype=xtype)
            else:
                # only take the predicted channels
                residual = x

        if comm.get_size("fin") > 1:
            x = scatter_to_parallel_region(x, 1, "fin")

        if self.checkpointing >= 1:
            x, res_cube, res_origin, res_mid = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x, res_cube, res_origin, res_mid  = self.encoder(x)
            
        if hasattr(self, "pos_embed"):
            if self.pos_embed.type == "frequency":
                pos_embed = torch.stack([self.pos_embed[0], nn.functional.pad(self.pos_embed[1], (1, 0), "constant", 0)], dim=-1)
                with amp.autocast(enabled=False):
                    pos_embed = self.itrans_up(torch.view_as_complex(pos_embed))
            else:
                pos_embed = self.pos_embed

            # add pos embed
            x = x + pos_embed

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)
        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x, res_mid, res_origin, use_reentrant=False)
        else:
            x = self.decoder(x, res_mid, res_origin)
        # print(x.shape, res_cube.shape)
        x = torch.cat([x, res_cube], dim=1)
        # x = self.cube_conv(x)
        x = checkpoint(self.cube_conv, x, use_reentrant=False)
        if hasattr(self.decoder, "comm_out_name") and (comm.get_size(self.decoder.comm_out_name) > 1):
            x = gather_from_parallel_region(x, 1, self.gather_shapes, self.decoder.comm_out_name)


        # big skip: learning diff
        if self.big_skip:
            x = x + self.residual_transform(residual)
        
        return x

@dataclass
class SphericalFourierNeuralOperatorNetSfnoEncMetaData(ModelMetaData):
    name: str = "SFNO_SFNOENC"

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True

SFNO_SFNOENC = modulus.Module.from_torch(
    SphericalFourierNeuralOperatorNetSfnoEnc,
    SphericalFourierNeuralOperatorNetSfnoEncMetaData()
)

