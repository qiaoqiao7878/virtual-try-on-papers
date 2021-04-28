# virtual-try-on-papers 虚拟试衣论文总结

2D虚拟试衣2021最新论文


最近呢，实习的公司让我研究一下最新的Virtual Try-On 虚拟试衣的论文，做个小汇报。所以一个星期看了大概三十几篇有关虚拟试衣的英语论文。



昨天看到一句话说，“你不需要是一个天才，才有资格分享知识。”，十分受触动。毕竟我不是这个领域的专家，也不是大牛，但初学者也可以分享给初学者或者介绍给完全不是这个领域的人了解嘛。



于是就想写一篇让我妈都能看得懂的最新虚拟试衣论文总结，主要介绍2021年新发表的5篇我认为效果最好的虚拟试衣论文。写之前号称我妈也能看得懂，写完之后发现还是有点难度。。。专业名词有点多，也很难一一解释，尽量做了一些通俗易懂的比喻和举例子。o((>ω< ))o



目前的虚拟试衣大致分为两大类，二维和三维。

二维的就是有一张用户照片，一张或几张服装的照片，合成一张用户穿着服装的虚拟试衣效果的图像。

三维的主要是把服装的款式、图案映射，渲染到三维模型上。



下图是其中一种3D的虚拟试衣，这篇论文假设人体模型，衣服模型都是给定的。这难以反映用户的真实形象，衣服的款式也有很大的限制。并且3D数据较难收集，耗时耗力。

![pix2surf](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/pix2surf.JPG?raw=true)

图片来自论文Learning to Transfer Texture from Clothing Images to 3D Humans



所以我主要关注的是2D的虚拟试衣。主要任务是在保持用户的脸、肤色、体型、姿势的前提下，合成用户穿着另一件衣服的图像。另一件衣服的款式、花纹、图案等等也要在合成的图像中很好的保留。



1. VITON-HD：High-Resolution Virtual Try-On via Misalignment-Aware Normalization



VITON是Virtual Try-On的缩写，HD是High-definition，高清。VITON-HD是我认为目前的SOTA论文， state of the art，技术发展最新水平。



![VITON-HD](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/VITON-HD.JPG?raw=true)

VITON-HD



最左侧的是原图，左下角的衣服是试穿的衣服，其他的则是合成的图像。



VITON-HD算是2018年的一篇论文VITON: An Image-based Virtual Try-on Network的延续。VITON这篇论文的影响力还是很大的，后续的比如CP-VTON,CP-VTON+,SP-VTON,LA-VTON,VTNFP,ACGPN等论文都有VITON的影子。这里就不一一介绍了。



VITON 数据集有上万张女模特的照片，正面视角为主，以及她们所穿的衣服图片。图片大小为256×192，清晰度并不是很高。而 VITON-HD成功合成出 1024×768的图像。它也为此收集了更高清晰度的数据训练集VITON-HD。



这里的所有论文都用到了Deep Learning深度学习，而且是Deep Learning的升升升级版，各种具体的神经网络我就不介绍了。但深度学习可以简单理解为：

比如，你想让电脑判断一张图片上是猫还是狗。你就给它海量的数据，猫或者狗的图片，以及每张图片附上一个标签：猫/狗。

神经网络训练好之后，你给它一张没有标签的图片，它自己就会给出答案。当然了，答案不一定100%准确。

同理也可以实现目标定位，判断物体类型，图像分割等等计算机视觉的任务。

![types](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/types.png?raw=true)


人工智能号称像人一样只能，但其实背后都是有超大量的数据采集，科学家工程师的研究及编程，开发不同的神经网络架构，数学的各种优化算法的支撑，日日夜夜地跑代码优化神经网络。



为了训练Deep Learning深度学习神经网络来实现虚拟试衣，理想的数据集应该是：用户+自己的衣服，要试穿的衣服产品图，用户+试穿的衣服。但这是非常难得到的。一般情况下，网站上只有产品图和模特穿着产品的照片。没有数据就无法



VITON类的论文的办法是，那就想办法把图片上衣服的信息去掉。



![VITON-HD_process](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/VITON-HD_process.JPG?raw=true)


VITON-HD流程图





首先，用已有的办法得到身体各个部位的segmentation map 分割图。用去掉手臂和衣服的分割图、表达身体位姿的Skeleton骨骼图和产品图，去生成一个目标位姿的分割图。右上角的橙色模块。

值得注意的是，分割图上表示手部的部分没有被去掉，因为手部非常难被很清晰的生成出来。

左下角的模块是用上述这些信息和之前预测的这个衣服应该变成什么样的轮廓去预测薄板样条插值 Thin plate spline transformation（TPS变换）的参数。通过TPS变换把产品图变成一个变形后的衣服。



最后再把所有这些信息整合到一起，生成最终的图像。因为TPS变换的自由度有限，通过算法变换过的衣服不可能百分百和真实照片中模特穿在身上的衣服重叠，所有这里会有misalignment错误对齐。

ALIAS *ALIgnment-Aware Segment normalization模块是VITON-HD提出来的，旨在去除misalignment错误对齐的信息的影响。具体就不介绍了。



![VITON-HD example](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/VITON-HD%20exampleJPG.JPG?raw=true)


VITON-HD与其他论文的对比



第一列为原图，第二列为产品图，最右侧的一列是VITON-HD合成的效果图。根据论文提供的图片来看，效果十分惊艳。既可以生成原本被衣服覆盖住的肌肤，衣服的图案，款式，花纹又都可以较好的复现。尤其是衣服的领口和袖子，与之前的方法相比有很大的提升。并且清晰地保持了原模特的脸部，手部，裤子等。



2. Disentangled Cycle Consistency for Highly-realistic Virtual Try-On

简称DCTON

![DCTON](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/DCTON.JPG?raw=true)


DCTON流程图

这篇论文的特殊之处在于用了自我监督式学习， self-supersivison的end-to-end端到端学习。两个CNN 卷积神经网络的encoder-decoder编码器解码器是一样的结构。Cycle Consistency training。不停地循环，比较生成图和原图，直到误差足够小。

![DCTON example](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/DCTON%20example.JPG?raw=true)

DCTON与其他论文的对比



可以看到，之前的许多论文，领口形状是跟随了原图，而不是产品图。而且当原图穿着宽松的袖子，而产品图是背心， 以前的方法会生成错误的手臂形状。而DCTON解决了这些问题，衣服复现效果也十分不错。





3. Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-on and Outfit Editing

简称DiOr



很有意思的一篇文章，思路与上面两篇完全不一样。实现的功能很多，包括上衣是否扎进裤子，多层叠穿（而且是无限叠穿），换姿势，甚至去除衣服图案，添加衣服图案等。而且要试的衣服可以是产品图或者模特图。

![DiOr](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/DiOr.JPG?raw=true)


DiOr

它采用encoder-decoder 编码器解码器，分别编码位姿，体型，头，手臂，每件衣裤的形状和外观，再一层层把他们解码成图像。通过改变上衣和裤子的先后顺序就可以改变最终上衣是否扎进裤子的效果。比如通过替换短袖的外观编码，就可以得到模特穿着不一样的短袖。替换位姿编码就可以得到摆着不同姿势的模特图。

![DiOr_process](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/DiOr_process.JPG?raw=true)




跟VITON类只能生成一个姿势的图像相比，这种方法用途多了不少。但它采用编码解码，复现能力有限，颜色或者相似的花纹没问题，但很难生成和产品图上完全一样的图案或者文字。

DiOr (Dressing in order)采用的是DeepFashion Datase。这个数据集里面有同一个模特不同位姿的照片。



4. Style and Pose Control for Image Synthesis of Humans from a Single Monocular View

简称StylePoseGAN

![StyleposeGAN.JPG](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/StyleposeGAN.JPG?raw=true)


StylePoseGAN

![StylePoseGAN%20overview.JPG](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/StylePoseGAN%20overview.JPG?raw=true)


StylePoseGAN流程图



用DensePose提取位姿pose和外观appearance。把位姿和外观编码成一个tensorz张量和一个vector向量。张量和向量可以理解为一系列数值。



用这些数值通过一个style-based generator(基于风格的生成器)合成最终图像。

可以实现换衣，换姿势，甚至换头的效果。图片也较为清晰。



5. VOGUE: Try-On by StyleGAN Interpolation Optimization



这篇论文的方法也很有意思。它首先训练一个pose conditioned（受限制于位姿）的 StyleGAN2.



GAN, Generative adversarial network,生成对抗网络。是非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。生成对抗网络由一个生成网络与一个判别网络组成。生成网络从潜在空间（latent space）中随机取样作为输入，其输出结果需要尽量模仿训练集中的真实样本。判别网络的输入则为真实样本或生成网络的输出，其目的是将生成网络的输出从真实样本中尽可能分辨出来。而生成网络则要尽可能地欺骗判别网络。两个网络相互对抗、不断调整参数，最终目的是使判别网络无法判断生成网络的输出结果是否真实。
维基百科
有种魔高一尺，道高一丈的感觉，我骗你，你判断我是不是在骗你。骗来骗去，两个人经验越来越丰富。直到你无法判断我说的是真是假。通过GAN，电脑可以学习如何生成以假乱真的图片。



StyleGAN 和 StyleGAN2也是很有启发性的论文。下图是电脑合成的人脸，只有最左一列和最上一行是真实的照片，其他都是合成的呦。既然它可以合成人脸，同样的原理，只要给足够多的数据，它也可以被训练用来合成模特试衣图。



![StyleGan.JPG](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/StyleGan.JPG?raw=true)


StyleGAN合成的人脸



这个受限于位姿的StyleGAN2可以根据二维的骨骼图2D pose skeleton生成一个穿着某套衣服的人，摆着与骨骼图一样姿势的图像。

然后把输入图像input image "project"投射到latent space潜在空间里。用户图像得到一个latent code隐编码, 衣服得到一个latent code, 那么我们希望生成的图像就在隐空间里的某个地方，是这两个隐编码的线性组合。要做的就是训练这个网络如何找到这个我们想要的线性组合。



![VOGUE](https://github.com/qiaoqiao7878/virtual-try-on-papers/blob/main/images/VOGUE.JPG?raw=true)


VOGUE



左中右分别为，用户，模特图，用户穿着模特身上的衣服或裤子。上面一行是换衣，下面一行是换裤子。

用户和模特体型相差较大也没关系，对纯色衣服非常适用，花纹也能模仿，能够生成512*512像素级别的图像，但也是很难生成和衣服上完全一样的图案或文字。



以上就是我看了30多篇虚拟试衣的论文之后挑出来觉得效果最好的5篇。



只是非常简单、浅层的总结介绍，希望对想简单了解的朋友有所帮助。感兴趣的朋友可以去看原文，新技术还是很impressive的。
