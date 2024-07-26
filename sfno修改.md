## sfno 修改

models/networks/sfnonet.py

class SpectralFilterLayer

由于encoder中输入维度和输出维度不一样，故增加一个参数if_encoder，对该类作是否属于encoder的分类，并在spectral conv中修改输入输出维度

![e59d797e158b8481af9e37bdb967b7d](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\e59d797e158b8481af9e37bdb967b7d.png)

![5f02a1e2931f5c79b299cda4c03c830](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\5f02a1e2931f5c79b299cda4c03c830.png)

![a1a76cff96980be01a6561514cd4ff3](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\a1a76cff96980be01a6561514cd4ff3.png)

class FourierNeuralOperatorBlock

修改原因和方式同上。

![961b20e91ead011c43dfb319254907b](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\961b20e91ead011c43dfb319254907b.png)

![7f420f33e20689945e8cc59f191999b](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\7f420f33e20689945e8cc59f191999b.png)

![7f420f33e20689945e8cc59f191999b](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\7f420f33e20689945e8cc59f191999b.png)

![6f8880c3db8292fdf53ccab59e43022](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\6f8880c3db8292fdf53ccab59e43022.png)

![b8e9fee3c8b1319cbb27e19055013f5](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\b8e9fee3c8b1319cbb27e19055013f5.png)

230行 参数初始化，sqrt中的分母选择我并无依据，还需斧正。

![507dbfd18309dd8c0f5b513a8f76595](C:\Users\jiaqi\Documents\WeChat Files\danny306779\FileStorage\Temp\507dbfd18309dd8c0f5b513a8f76595.png)



718行起，到800行，class Encoder_sfno，为单个变量的encoder。

801行起，到890行，class EncoderWrapper_sfno，为整个encoder，包含

892行起，到文件末，class SphericalFourierNeuralOperatorNetSfnoEnc，为替换encoder后的模型。