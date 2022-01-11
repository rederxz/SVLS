**问题定义**

- 从label的角度做文章，目标是改进model calibration，次级目标是涨点



**目前的想法**

- 把label作为一个分布进行采样，即在线的数据增强
  - 或者把voxel进行变形，然后label不变？
  - label smoothing - random sample - voxel deformation
- 按照动态轮廓的能量函数去评价各个patch，采取不同的blur程度
- 添加水平集深度损失
- 对label进行膨胀和腐蚀操作，进行在线的数据增强
- 考虑当前像素离最近边界的距离
- 考虑对原label进行几何运算，如果之后还能保留，则不进行soft， 一个依据就是要尽可能保留小的突起，防止被低通滤波滤掉
- 或者是结合SVLS和level set，参考17年文章，让模型预测水平集（以水平集作为soft label），区域边缘的像素的标签soft，区域中心的像素的标签hard
- 可以这样解释为什么要使用level set，因为CNN本身的特点导致无法对边缘精确预测（具体原因是什么看水平集那篇文章）
- 利用层之间的相似性，进行层之间的配准，然后比较label的差异，得到不确定性
- 采用基于patch的纹理或相关方法来获得不确定度
- 直接对onehottrain出来的模型输出做blur，作为最终结果？
- 我猜他的最大亮点就是简单，我们一旦把这个东西搞复杂了价值就小了
- 做类别间的n-d灰度直方图（联合灰度直方图），直方图均衡化
- 根据微信那篇推送，选择一个三维块，根据方差进行label smoothing，方差大则表示不确定性更强
- 能不能用上互信息？LC^2 MIND local structure descriptor? 图像融合？看作一个配准问题，评估配准的程度？
- 千方百计找自己方法的优点
- 可以理解SVLS本身的特点：认为分割label本身可以标志真正的边界，在边界处voxel的标签是不可靠的；我们认为可以通过图像本身的特点，获取到voxel的不确定性

- 新的对抗式训练：
  - 生成器：根据原始label生成soft label，想让loss变大
  - 分割器：分割，想让loss变小

- **解决这些问题**：
  - 如何评估生成的soft label的合理性？第一步实际上就是由one-hot得到soft label，那么如何评估这一步的结果？
  - 如何保持小的结构？目前SVLS对于小结构是不友好的，因为其本质是一个低通滤波，如果一个结构本身就很小，那么经过低通滤波后，就可能被滤掉；先super pixel，然后平均一个superpixel内部的标签？或者进行super pixel，然后对一个super pixel内部进行相对距离的soften？或者直接提取label map的边界，以每个块进行相对距离soften？
  - 目前SVLS中“临近及易混淆”的想法是否合理？得到soft label的方式仅仅是基于临近的voxel，然而离得近并不一定容易混淆，或许可以使用super pixel进行分块，然后得到一个伴生矩阵？（对某一类，其他类和它出现在同一个super pixel的概率）目前SVLS进行平滑的地方是否足够？是否可以先进行super pixel，然后边界处的label也要进行soft？
  - 是否能够利用图像的信息？这里可以考虑一下label的边缘位置，如果图像中这里存在很强的梯度，就认为这个边界是可靠的？否则就要进行一定程度的平滑？由于存在多种序列，能否根据序列之间的一致性对标签的可靠性进行评判？
- 使用kl散度进行训练？



**目前的一个计划**
Story:

1. 现在的calibration相关工作在multi-rater的情况下取得较好进展，但是multi-rater的label数据不易获得，
   导致这些工作无法得到更加广泛的应用。为此我们提出一种新的范式，从single-rater的label中获得"multi-rater"的label，从而可以
   使用将这些工作的方法拓展到single-rater的场景中。
   
2. SVLS是将LS应用于医学图像分割的很好尝试，但是其也存在问题，比如造成label的高频信息丢失等。我们认为，更加合理的blur方式并不是
   基于临近的其他label，而是基于混淆或不确定度的方式。对于multi-rater，这种混淆或不确定度通常是容易定义并且容易获得的，但是对于
   single-rater并非如此。为此，我们定义了一种衡量single-rater标签不确定度的方式，并设计了一套流程对其进行提取。基于这种不确定度的
   LS使得我们得到了更好的分割效果和model calibration。 


Phase 1:
使用2D的分割进行训练，对层片进行随机抽样，获得几组slice，分别训练，然后交叉推理，比较output，获得不同voxel的不确定度
这个过程用于捕捉intra-rater variability
Phase 2:
利用不确定度获取soft label，使用soft label为目标，进行3D分割训练



**我们需要做：**
需要调研：

- 有没有已有的工作去估计single rater的label的不确定度(intra-rater的不确定度)？
- 如何复现文中关于model calibration的结果？（目前复现的只有关于dice的结果）


在展示中我们需要：
- 把这个问题定义好（讲清楚我们的最终目的及临床意义）
- 梳理我们的思路是啥（label smoothing-uncertainty-获取uncertainty）
- 工作计划，最终可能的成果



**王老师建议：**

- 在得到label后进行可视化，验证方法的效果是否符合预期
  - 我们最后弄出来标签很soft的地方，到底是不是容易标错的地方或者multi-rater情况下不一致性大的地方
- 控制好时间，课程结束的时候要有阶段性成果
- 做着做着想法就有了



**反思**

- 不同数据集可能适合不同的smoothing模式，比如肿瘤分割，边缘不规则，适合偏向于oh；大的器官分割，边缘较规则，适合偏向于svls？

