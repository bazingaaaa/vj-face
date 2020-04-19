Viola jones face detection implementation
====
### 说明
语言：c/c++  
平台：macOS/linux  
环境：opencv4.2.0 + openmp  



参考文章
======
1. [An Analysis of the Viola-Jones Face Detection Algorithm]( http://www.ipol.im/pub/art/2014/104/article.pdf)

2. [Robust Real-Time Face Detection](https://www.face-rec.org/algorithms/Boosting-Ensemble/16981346.pdf)

3. Rapid Object Detection using a Boosted Cascade of Simple Features

   


示例
=====







haarcascade框架使用说明
====



### 训练阶段 

第一个参数: train，代表进行训练 
第二个参数: 模型训练时用到的数据文件路径（下面有数据格式示例） 
-model模型配置文件路径，模型支持继续训练，如果需要在先前模型基础上继续训练，可给出先前模型文件路径（可选） 

### 测试阶段

第一个参数: test，代表进行测试  
第二个参数: 测试文件路径，一般为*.jpg, *.png, *.gif
-model  模型配置文件路径（可选，默认加载模型./backup/attentional_cascade_def.cfg）  
-outfile  测试得到的文件保存路径，可通过opencv直接查看，编译时需配置opencv选项（可选） 
-skintest 肤色测试开关，1打开  0关闭，测试彩色图像时可以打开，测试灰度图像时需要关闭，默认为关闭（可选） 

### 训练数据示例

    window_size = 24  (检测窗大小，同样也是正样本的图片大小，必须是正方形)
    train_positive = ./train_pos.list （./train_pos.list文件中包含了所有训练集中的正样本路径，列表形式）
    train_negative = ./train_neg.list （./train_neg.list文件中包含了所有训练集中的负样本路径，列表形式）
    validation_positive = ./vali_pos.list（./vali_pos.list文件中包含了所有验证集中的正样本路径，列表形式）
    validation_negative = ./vali_neg.list（./vali_neg.list文件中包含了所有验证集中的负样本路径，列表形式）
    backup = ./backup （模型训练时的保存路径）
    fnr_perstage = 0.005  (单个阶段内的假阴性率训练目标)
    fpr_perstage = 0.5   (单个阶段内的假阳性率训练目标)
    fpr_overall = 0.0000001  (整体模型的假阳性率训练目标)

训练时用到的正负样本需为灰度图像，且每个正样本图像大小为 window_size x window_size像素，负样本图像大小可以不固定

可参考：[Viola jones face detecion原理]()

