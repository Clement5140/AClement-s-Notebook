# Planning and Decision-Making for Autonomous Vehicles

## 1.介绍

主要介绍自动驾驶规划和决策的多个方面，大致分为三种不同的方法：  
sequential planning(section 2)，behavior-aware planning(section 4)和end-to-end planning(section 3)。  
section 5讲各种方法如何验证和综合，section 6讲自动驾驶车辆车队(fleet)的管理方法。

## 2.MOTION PLANNING AND CONTROL

主要是一些车辆控制和运动规划的传统方法。  
先讲parallel autonomy，再讲autonomous vehicles，最后列出决策和规划目前的一些挑战。

### 车辆动力学和控制

低速情况下，有运动学模型可以控制车辆。而高速情况下，需要使用车辆的完整动力学模型，包括轮胎的力等。  
一些模型：Nonlinear control (18), model predictive control (19), feedback–feedforward control (20)

这些模型依赖于需要识别的车辆模型，有基于优化的技术和基于学习的技术。

### parallel autonomy

有三种collaborative autonomy的方式：series autonomy，interleaved autonomy，parallel autonomy。  
parallel autonomy指自动驾驶系统处于后台运行来保证安全，而由人来操作车辆，当人被转移注意力或无法处理当前情况时，该系统会提供安全性。

比较直观的做法是将人的输入和安全系统的输出直接线性组合起来，Anderson et al. (23)。

另一种做法是将人的输入以最低影响的方式合并到优化框架中。Alonso-Mora et al. (25)，Shia et al. (26) ，Erlien et al. (27)。

### 运动规划

大多数计算安全路径的传统方法都是基于以下三种思路：

* 支持碰撞检测的输入空间离散化(input space discretization with collision checking)
  * lattice planners (e.g., 31, 32)
  * road-aligned primitives (e.g., 33)
* 随机规划(randomized planning)
  * rapidly exploring random trees (RRT) (e.g., 34, 35)
* 约束优化和滚动时域控制(constrained optimization and receding-horizon control(e.g., 19, 36))
  * Schwarting et al. (28)

## 3.INTEGRATED PERCEPTION AND PLANNING

主要介绍感知的前沿技术，描述综合感知和规划的端到端方法(end-to-end methods)，直接从感知信息生成车辆的控制输入，非常依赖机器学习。

### 从经典感知到基于神经网络的感知系统的目前的挑战

一些benchmarking数据集：KITTI (42), ISPRS (International Society for Photogrammetry and Remote Sensing), MOT (Multiple Object Tracking), and Cityscapes (43)。

经典感知系统从原始感知数据中以人工设计的特征形式提取信息。最著名的一些例子：SIFT (Scale-Invariant Feature Transform) (44, 45), BRISK (Binary Robust Invariant Scalable Keypoints) (46), SURF (Speeded Up Robust Features) (47, 48), and ORB (Oriented FAST and Rotated BRIEF) (49, 50)。  
基于纯视觉的迅速、轻量级的方法已经成熟：such asORB-SLAM2 (50), SVO(SemidirectVisualOdometry) 2.0 (52), and LSD-SLAM (Large-Scale DirectMonocular SLAM) (53)。

物体检测一般有两种方式：碰撞盒检测和语义分割。

* 碰撞盒检测
  * the ImageNet Large Scale Visual Recognition Challenge (55)
  * real-time-capable systems such as Faster R-CNN (Faster Regional Convolutional Neural Network) (56)
* 语义分割
  * ResNet38 (57) and PSPNet (Pyramid Scene Parsing Network) (58)
    * achieve more than 80% mIoU (mean intersection over union) in the Cityscapes data set (43)
    * but take multiple seconds to propagate on high-resolution images
  * ENet (Efficient Neural Network) (59)
    * achieved a 13-ms runtime on 1,024×2,048–pixel images with 58% mIoU on theCityscapes data set (43)
  * ICNet (ImageCascadeNetwork) (60)
    * achieved 70% mIoU at 33 ms

真实世界数据集非常昂贵，使用虚拟世界的数据更便宜，训练效果也更好，不过会增加数据集的偏差。

基于神经网络的感知系统有一个很大的问题，不确定性的反馈不足(insufficient feedback of uncertainty)。
网络不确定性可以由Monte Carlo dropout sampling (65)估计。
McAllister et al. (66)提出使用有原则的贝叶斯框架(a principled Bayesian framework)估计和传播整个系统管道中每个组件的不确定性，将使自动驾驶汽车能够适当地应对高不确定性。

### 端到端规划(End-to-End Planning)

传统的自动驾驶架构中，功能被封装在模块之间清晰可见的接口中，也被称作中介感知(mediated perception)。

另一种架构是对感知模块的某些部分进行训练来包含规划模块的部分任务。

* Caltagirone et al. (68)，通过整合激光雷达点云、GPS-惯性测量单元（IMU）信息和谷歌导航信息来生成行驶路径。
* 语义分割网络可用于在相机图像空间(camera image space (69))中生成路径。

更进一步，架构可以学习车道和道路跟踪的整个任务，而无需手动分解为道路或车道标记检测、语义抽象、路径规划和控制。

* ALVINN (Autonomous Land Vehicle in a Neural Network) (70)，训练神经网络从相机图像输出行驶的转向角，来让车辆保持在道路上行驶。
* Chen et al. (67)将其称为behavior reflex approach。
* NVIDIA (72)，训练了一个深度卷积神经网络，可以将前向摄像头的原始(raw)图像直接映射到转向命令，并能够处理具有挑战性的场景。
* Bojarski et al. (73)，展示了神经网络能够学习类似车道标线、道路边界和其他车辆形状的特征(feature)。
* Xu et al. (75)，使用大规模行驶视频数据集来训练了一个端到端的全卷积LSTM神经网络，可以预测离散行为(直行、停止、左转、右转)和连续行为(方向盘角度控制)。
* SafeDAgger (76)，DAgger (77)。

端到端运动规划也被运用于机器人学。

另一条研究路线是在模拟器中学习驾驶行为，可以在安全环境下观察失败的情况，适合强化学习的训练。

* Wolf et al. (81)，在模拟环境下使用Deep Q-Network来学习驾驶车辆。

## 4.BEHAVIOR-AWARE MOTION PLANNING

## 5.VERIFICATION AND SYNTHESIS

## 6.FLEET MANAGEMENT
