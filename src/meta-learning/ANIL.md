# RAPID LEARNING OR FEATURE REUSE? TOWARDS UNDERSTANDING THE EFFECTIVENESS OF MAML

文章目的是探讨MAML有效的原因是可以快速学习还是特征重复利用，前者模型在遇到新的任务时会进行大量、有效的修改，而后者是说模型在训练之后已经包含了相关任务高质量的特征。

文章直接给出结论：MAML有效的原因是特征重复利用（feature reuse），通过layer freezing experiments和分析MAML模型latent representations来证明结论，并在结论上提出MAML的简化版本ANIL。

## FREEZING LAYER REPRESENTATIONS

作者提出模型的网络需要分为两个部分：head和body，head指网络最后一层，body指前面其余的层。

在每个few-shot任务中，网络最后一层也就是head需要将神经元和类别对应起来，所以不同的任务inner-loop训练后head层会有较大不同，所以只需要主要关注body层的表现。

layer freezing experiments指在测试时禁止网络的body层的一个连续子集的参数更新，来对比这些结果，比如：网络一共四层，记为1、2、3、4，最后还有一个head层，实验会进行五次，第一次不禁止更新，第二次禁止第1层更新，第三次禁止第1、2层更新，以此类推。

结果显示对参数禁止更新几乎不影响模型的准确率，即使禁止body中的所有层更新参数，模型的表现也没有怎么下降。

## REPRESENTATIONAL SIMILARITY EXPERIMENTS

这个实验分析神经网络在每个任务的inner loop适应后latent representations有多少改变。

作者使用了CCA相似度和CKA相似度来对网络的每一层在inner loop前后的不同进行测试，CCA提供了一种方法可以比较神经网络中的两个层的表示的相似度(从0到1，分数越接近1越相似)，结果发现网络body的所有层相似度都大于0.9，只有head层小于0.5，也符合之前的想法。

## ANIL(Almost No Inner Loop)算法

在ANIL算法中，作者移除了训练和测试时inner loop对网络body层的更新，只保留对head层的更新，其余和MAML算法一样。

由于ANIL算法几乎没有inner loop，其速度有显著提升，并且：

* 模型的表现能够达到MAML的水平，不管是few-shot图像分类还是强化学习场景下。
* ANIL和MAML训练时的loss和acc曲线几乎一致，使用CCA和CKA相似度发现MAML-ANIL表示和MAML-MAML表示以及ANIL-ANIL表示有同样的平均相似度得分，说明两个算法学习的特征是类似的，训练时有没有inner loop更新不会改变学习到的特征种类。

## 网络head和body的贡献

好的特征已经被学习到的情况下，测试时网络的head有多重要？

### NIL(No Inner Loop)算法

在训练之后，好的特征已经被学习到了，探讨此时(测试时)head的重要性。

NIL算法：

* 1.使用ANIL/MAML算法训练一个few-shot模型，作者使用了ANIL。
* 2.测试时，移除训练模型的head，对于每个任务，先将k个有标签的数据(支撑集support set)传入网络的body，得到他们倒数第二层的表示，然后对于一个测试数据，计算他倒数第二层的表示和支撑集的表示的余弦相似度(cosine similarities)，使用这些相似性来加权支撑集的标签。

NIL算法得到的模型性能和ANIL和MAML类似，说明MAML/ANIL训练得到的网络body学习到的特征是最重要的。

### 网络body的训练方式

NIL又引出一个问题：训练时head和task alignment的重要性。

训练时使用NIL方法，即没有head，得到的模型表现会降低很多，说明训练时head是很重要的。
