viola jones人脸检测原理
=====

VJ人脸检测算法是一种基于滑动窗口的目标检测算法，但它却克服了滑动窗口检测带来的低效问题，可以用于实时人脸检测，主要归功于以下三点：

1. 利用一种新的数据结构”积分图像“来快速计算Haar-like特征
2. 利用adaboost对候选特征进行筛选，找出具有更强分辨力的特征
3. 利用attention cascade将更多更为细致的计算应用至感兴趣的区域

下面进一步详细介绍整个检测原理。

## 基于滑窗的目标检测

基于滑窗的目标检测基本原理很简单，首先构建一个detector（检测器），以人脸检测为例，检测器的工作就是判断给定大小的图像是否是一张人脸，再用该检测器从左至右从上到下扫描各个子窗（检测器扫描的部分图像称为子窗，文章中用到的窗口大小为24x24像素），当检测器判断某个子窗包含人脸时，即完成了人脸检测。

此处有图。

这样处理有个问题，如果图像中包含的人脸变大了，此时用固定大小的滑窗就无法进行检测。通常有两种解决方法，1. 采用image-pyramid（图像金字塔），也就是通过resize获得多种不同大小图像并堆叠在一起，用固定大小检测器同时对所有图像进行扫描，2. 采用不同大小的检测器对图像进行扫描。文章中用到的是第二种方法，尽管如此，虽然避免了调整图像大小带来的计算开销，但不同大小的检测器意味着有更多子窗等待检测。

如何构建一个足够快的检测器。

## Haar-like特征





## AdaBoost算法





## Attention cascade





## 检测后处理







## 算法不足之处







参考文章
-----

1. [An Analysis of the Viola-Jones Face Detection Algorithm]( http://www.ipol.im/pub/art/2014/104/article.pdf)
2. [Robust Real-Time Face Detection](https://www.face-rec.org/algorithms/Boosting-Ensemble/16981346.pdf)
3. Rapid Object Detection using a Boosted Cascade of Simple Features

