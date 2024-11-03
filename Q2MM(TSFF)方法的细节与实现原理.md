# Q2MM(TSFF)方法的细节与实现原理

TSFF力场由来已久，早期的科研人员通过传统的分子力场方法直接应用到反应过渡态的方式进行反应动力学研究。Q2MM是一种经过Per-Ola Norrby课题组经过长达二十多年开发形成的全新具有高度实用价值的过渡态力场方法，目前已经被集成到ACE（Asymmetric catalytic evaluation）和CatVS（Catalyst virtual screening）这两个全自动化均相催化剂虚拟筛选软件之中，相关论文发表于Nat. Catalysis，JACS，JCC，Chem. Sci. 等期刊。

![1c07abea3fa047817db72a78c606ade](E:\Softwares\WeChat\WeChat Files\wxid_mw2n7uz79zg222\FileStorage\Temp\1c07abea3fa047817db72a78c606ade.png)

TSFF（过渡态力场，Transition State Force Field）方法是专门为准确预测化学反应中的过渡态能量而开发的。传统的力场方法（如分子力学）主要用于预测分子在平衡状态下的几何结构和相对稳定的构象转变。然而，传统力场通常不适用于反应中的键断裂或形成过程，因为这些过程涉及显著的几何变形和势能面的复杂变化。

TSFF方法的核心创新在于，将过渡态视为势能面的局部极小值，并通过重新构建力场参数，使得过渡态几何结构可以通过最小化能量直接获得(2003_TSFF过渡态力场的原理)。与常规力场方法不同，TSFF不依赖反应物和产物的势能面，而是直接将过渡态的几何信息融入力场中。这种方法不仅能够有效捕捉过渡态的能量，还能准确预测类似反应的相对反应性和选择性。

## TSFF的原理

### 力场参数的回归流程

![image-20241030101850260](C:\Users\TangKun\AppData\Roaming\Typora\typora-user-images\image-20241030101850260.png)

### 过渡态结构优化上的巧思

TSFF拟合需要QM计算得到的精确过渡态的结构以及其对应的Hessian矩阵，这里对标签值中的Hessian矩阵进行了修改，将Hessian矩阵对角化之后得到的特征值中的唯一的负值改成了+1（一个相对较大的正特征值），这样做会使得过渡态沿着反应路径（MEP）的势能面的曲率反转变成正值，从可视化角度看就是势能面进行了翻转，如下图所示，原先过渡态上的鞍点变成了极小点，这样操作之后，就可以采用最小化优化算法来优化过渡态结构了。

![image-20241030102428789](C:\Users\TangKun\AppData\Roaming\Typora\typora-user-images\image-20241030102428789.png)

<!--补充知识1：E对R求二阶偏导数得到维度为(3*N，3*N)的Hessian矩阵，Hessian中的每一个元素都是关于两个不同原子坐标的能量的导数，其中对角元素是每个原子自身坐标关于能量的二阶偏导数，因此在使用Hessian时一般只需要上或者下三角阵即可包含全部所需的信息-->

<!--补充知识2：Hessian对角化操作可以求出特征值和特征向量，特征值对应分子系统中原子与原子之间的振动频率，特征向量对应振动方向及振幅 。特征值全部为正代表曲率为正，坐标不论怎么改变，不论向着什么方向，能量都会升高，这对应着势能面上的谷底（极小点）；特征值除了反应坐标方向为负值（曲率为负值，也是唯一的虚频的来源，参见下面的公式），其余都为正，这对应着过渡态，也就是势能面上的鞍点，因为这个点上的结构只有沿着反应坐标变化能量才会降低，而其他所有方向都会升高-->

<!--振动频率ω可以通过下面的公式计算，其中λ是特征值，μ是约化质量。-->
$$
\omega=\sqrt{\frac{1}{\mu}\lambda}
$$

### Hessian标签数据处理方式

* 首先通过QM做频率分析计算得到Hessian矩阵

* 对Hessian矩阵进行对角化得到特征值特征向量

* 将唯一的负特征值改为(+1)

* 根据修改之后的特征向量和特征值还原出新的Hessian

* 以新的Hessian拟合力场参数
  $$
  H=X^TDX
  $$

  $$
  \mathsf{H}^{\prime}=\mathbf{X}^{\mathsf{T}}\mathbf{D}^{\prime}\mathbf{X}
  $$

  $$
  \mathbf{D}=\begin{bmatrix}-\lambda_1&&\cdots&&\mathbf{0}\\\\&\mathbf{0}&&&\\\\\vdots&&\lambda_\mathbf{8}&&\vdots\\\\&&&\ddots&\\\\\mathbf{0}&\cdots&&&\lambda_N\end{bmatrix}\quad\mathbf{D}^{\prime}=\begin{bmatrix}+\mathbf{1}&&\cdots&&\mathbf{0}\\\\&\mathbf{0}&&&\\\\\vdots&&\lambda_\mathbf{\Phi}&&\vdots\\\\&&&\ddots&\\\\\mathbf{0}&\cdots&&&\lambda_N\end{bmatrix}
  $$

```python
import numpy as np
# 对角化Hessian矩阵
eigenvalues, eigenvectors = np.linalg.eig(hessian_matrix)
```

## TSFF的发展以及拟合的注意事项



### 关于Hessian的拟合

TSFF拟合过程中惩罚函数是预测值与标签值之间的平方误差之和，最开始的工作中为了区分分别与键角、扭转、长程作用相关的Hessian元素，对惩罚函数中的平方误差和采用了加权因子缩放[Hagler and coworkers]。后续的更新版本TSFF去掉了加权因子缩放操作，直接拟合原始Hessian。

连接同一原子上两个坐标的Hessian元素，块对角元素，并不直接对应于力场相互作用，而是包含冗余数据。块对角线元素是非对角线元素的总和，并且倾向于将适当拟合的Hessian元素与可用分子力学函数无法表示的其他元素耦合。因此，通过从参数化过程中排除块对角元素，可以在不损失数据的情况下改进Hessian拟合，称之为“非对角线Hessian拟合”。

## 如何采用NFF取代TSFF



## NFF取代TSFF进行催化剂虚拟筛选的尝试