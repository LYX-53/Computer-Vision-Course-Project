## 基于改进扩散模型的文本到视频生成与编辑系统

​	视频生成与编辑作为计算机视觉和多媒体领域的核心任务，在影视制作、虚拟现实、教育娱乐等场景中具有广泛应用前景。传统方法如生成对抗网络（GAN）通过对抗训练实现了视觉内容的生成，但存在生成稳定性差、细节质量不足以及编辑可控性有限等问题，尤其是在处理多帧视频的时序一致性时，容易出现运动不连贯或模式崩溃。近年来，扩散模型在图像生成领域展现出卓越的性能，通过渐进式去噪过程实现了高保真样本生成，但其在视频领域的扩展面临时间一致性建模和计算复杂度高等挑战。

​	本研究针对现有方法的局限性，提出一种改进的扩散模型框架，通过引入指数移动平均（EMA）、混合损失函数、自适应组归一化（AdaGN）等创新模块，提升视频生成的稳定性和细节质量。同时，设计交互式视频编辑工具，实现尺寸调整、水印添加、帧率控制等功能，旨在构建兼具高质量生成与灵活编辑能力的一体化解决方案。

### 一、数据集

####1.1MSR-VTT与Shutterstock视频标注数据集

从Kaggle和HuggingFace下载MSR-VTT数据集，将视频转换为GIF格式，并为每个视频创建对应的字幕文本文件。为了构建高质量的文本到视频生成模型，我们需要多样化的带标注视频数据集。MSR-VTT（Microsoft Research Video to Text）是理想选择，它包含来自20个类别的10,000个视频剪辑，每个剪辑都有对应的英文描述标注，数据规模约为10,000，，含英文描述标注。数据下载，训练数据准备，为便于训练，对其进行预处理，将MP4转换为GIF格式，并创建对应地文本描述文件，构建文本视频对用于条件生成。经过处理将训练数据设置为video.gif-video.txt的结构

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps6.jpg) 

 MSR-VTT与Shutterstock视频标注数据集展示

####1.2Moving Mnist数据集

Moving MNIST 是用于评估时序模型的经典数据集，本研究直接采用其预处理后的序列数据。该数据集包含手写数字在网格中的运动序列，原始格式为 numpy 数组（维度为(序列长度, 批量大小, 高度, 宽度)）。为适配模型输入，首先将单通道灰度图像转换为 RGB 三通道，通过重复像素值实现维度扩展（如gray_img[:, :, np.newaxis].repeat(3, axis=2)）。其次，将数组转换为 GIF 格式，利用ImageSequenceClip工具从 numpy 数组直接生成视频帧序列，确保时序连贯性。由于 Moving MNIST 缺乏复杂语义标注，统一为所有视频分配简单文本描述（如 “digit 5 moving left”），聚焦于模型对动态物体运动轨迹的生成能力评估。该数据集的纯净性（仅包含数字运动）便于隔离验证模型的时序建模能力，排除复杂场景干扰。

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps7.png)![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps8.png)![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps9.png)![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps10.png) 

​         数据集展示

####1.3自行收集和设计数据集

#####1.3.1通过⽹络下载及爬取部分数据集

通过 GIPHY 网站（https://giphy.com/）搜索关键词（如 “cat playing”“sunset beach”“crowd walking”）获取多样化 GIF 资源，利用浏览器插件 ImageAssistant 批量爬取页面中的 GIF 文件。共收集到 2500+ 个 GIF，覆盖动物、自然、人群等场景。爬取后进行去重和质量筛选，删除分辨率低于 32×32 像素或帧率不稳定的样本，此部分数据补充了公开数据集在长尾场景的覆盖不足，增强模型对开放域内容的泛化能力。![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps11.jpg)

​                                    图1:    网站关键词搜索示例

 

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps12.jpg) 

​                                                 图2：  插件爬取过程 

#####1.3.2自行设计数据集

自有数据集可按需设计数据的关键特征，如时序复杂度、视觉属性、多模态标注，自有数据集可针对性地覆盖边缘情况（如低分辨率、运动模糊、遮挡），迫使模型学习更鲁棒的特征表示，增强模型的泛化能力；可紧密围绕研究问题设计，避免公开数据的 “无关噪声” 干扰实验结果。例如，研究 “基于文本的手势视频生成” 时，自行拍摄包含明确手势 - 文本对的数据集，可直接评估模型的语义对齐能力，而非被通用场景的冗余信息分散注意力，提高实验目标的精准性，避免因依赖公开数据导致的创新性被低估

我们用相机在校园内拍摄人物、植物、动物等类别的视频，要保证视频的稳定性、清晰度以及画面主体性，由于校园内可拍摄素材较少，将拍摄范围扩大到了商场、广场等地方，以提高数据集的多样性，收集好视频后进行剪辑，删除不稳定的画面，保持画面主体的连贯性以及裁剪视频的大小，再压缩处理成符合模型处理的大小，并用代码调用模型设计对应描述性文件，结构符合video.gif-video.txt一一对应的关系，最后整理成自有数据集，规模为100+。   

以下为gif预览图

  ![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps13.png) ![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps14.png)![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps15.jpg)

​     图3                                                    图4                                                 图5

图3描述性文件内容为a cat lying on the ground

图4描述性文件内容为a cat sitting on the ground

图5描述性文件内容为three man walking on the road

通过网络搜集和自行拍摄，小组设计了一份规模为2000+的数据集

| 数据集         | 规模   | 环境 | 核心场景       | 标注类型         | 用途                         |
| -------------- | ------ | ---- | -------------- | ---------------- | ---------------------------- |
| MSR-VTT        | 10000+ | A100 | 通用生活场景   | 英文文本描述     | 训练文本视频生成基础模型     |
| Moving MNIST   | 2000+  | L4   | 数字运动序列   | 简单动作标签     | 评估时序连贯性与动态建模能力 |
| 网络爬取数据集 | 2500+  | A100 | 开放域GIF      | 标签提取         | 增强模型泛化性，覆盖长尾效应 |
| 自行拍摄数据集 | 100+   | A100 | 可控人物或场景 | 调用模型撰写文本 | 验证语义对齐精度与编辑可控性 |

​                                                                     表1数据集对比

###二、基于改进的扩散模型的视频生成

本研究复现的扩散模型为：(https://avoid.overfit.cn/post/88567712b4f547469d74113f6d0810e0)

本研究改进的扩散模型架构如下：

####2.1 核心模型设计

#####2.1.1指数移动平均（EMA）集成

为提升训练过程的稳定性，模型引入指数移动平均（EMA）机制。通过维护独立的 EMA 模型副本，其参数由原始模型参数的指数加权平均更新，公式为：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps16.jpg)其中，decay为动态衰减率，初始值设为较低值（如 0.9），随训练步数增加逐步趋近于预设最大值（如 0.9999），更新频率由ema_update_every参数控制。在采样阶段，支持切换使用原始模型或EMA 模型：EMA 模型通过平滑参数更新轨迹，生成结果更稳定，适合高保真生成任务；原始模型响应速度更快，适用于实时交互场景。此设计有效缓解了扩散模型训练初期的波动问题，同时通过双模型机制平衡了生成质量与推理效率。

#####2.1.2混合损失函数

为优化生成视频的细节质量，设计混合损失函数，融合像素级重建损失与高层语义采用损失结合L1/L2损失与感知损失； L1/L2损失优化像素级重建精度； 感知损失利用预训练VGG网络提取多层特征，提升结构与语义一致性。

L1/L2 损失：直接优化像素级重建精度，计算公式为：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps17.jpg)其中，![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps18.jpg)为带噪视频帧，![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps19.jpg)为去噪网络，通过最小化预测噪声与真实噪声的距离，确保生成结果的像素级逼真度。

感知损失：利用预训练 VGG 网络（如 VGG16-BN）提取多层特征（如第 3、8、15、22 层），计算生成视频与真实视频在特征空间的差异，公式为：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps20.jpg)其中， ![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps21.jpg)VGG 第l层特征提取器，![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps22.jpg)为模型预测的初始帧。感知损失通过约束高层语义特征的一致性，提升生成内容的结构合理性与语义对齐度。

损失融合：通过超参数![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps23.jpg)动态平衡两类损失，最终损失函数为：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps24.jpg)

#####2.1.3自适应组归一化（AdaGN）

为增强时间步嵌入与条件信息的融合能力，设计自适应组归一化（AdaGN）模块融合时间步嵌入与条件信息（如文本编码），通过可学习比例因子增强特征交互，动态调节归一化组数，适应不同输入维度的特征分布，其核心结构包括：

时间步嵌入处理：通过 MLP 网络（SiLU 激活函数 + 线性层）将时间步嵌入映射为维度为![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps25.jpg)缩放 - 偏移向量，经可学习参数time_scale加权后得到time_scale_bias。条件信息处理：若输入条件（如文本编码、动作标签）存在，通过并行 MLP 网络处理为同类缩放 - 偏移向量，并与时间步向量相加，形成联合调节信号scale_bias。归一化与特征调节：先对输入特征图应用组归一化（GroupNorm），再通过scale_bias拆分为缩放因子scale与偏移量bias，实现像素级特征调整：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps26.jpg)其中，归一化组数可动态调节，以适应不同维度的输入特征分布。该模块通过可学习参数增强了时间依赖与条件语义的交互效率，尤其在文本引导的视频生成中显著提升了语义对齐精度。

####2.2 采样与推理框架

#####2.2.1. 多时间步采样（DDIM加速）

引入DDIM采样算法，支持快速生成（减少采样步数），并添加进度条控件提升可观测性。 返回预测初始帧（x_start）选项，用于中间结果分析。

模型集成DDIM（Denoising Diffusion Implicit Models）采样算法，支持通过参数ddim_num_steps自定义采样步数（如从标准 1000 步缩减至 20 步）。通过逆序时间序列![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps27.jpg)（S为缩放因子）减少迭代次数，并引入进度条控件实时反馈采样进度，增强可观测性。此外，采样过程中支持返回预测初始帧![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps28.jpg)，公式为：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps29.jpg)该中间结果可用于生成过程分析或迭代优化，为交互式编辑（如局部重生成）提供了基础支持。

#####2.2.2稳定性增强机制

clip_x_start`参数控制预测值截断范围，防止数值不稳定，提升模型鲁棒性。其中数值截断控制通过clip_x_start参数对预测初始帧进行动态截断。具体而言，基于训练数据的分位数（如 90% 分位值）计算截断阈值s，将预测值限制在[-s, s]范围内并归一化，公式为：![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps30.jpg)该操作有效抑制了极端值导致的数值不稳定，提升了生成结果的可靠性。其次类型提示与状态管理通过 Python 类型注解如torch.Tensor标注输入输出类型，增强代码可读性与鲁棒性，同时优化模型状态字典的加载保存逻辑。例如通过ema_update_count记录 EMA 更新步数，确保训练中断后可从最近稳定检查点恢复，避免重复训练开销。

改进后的模型结构如下：
![improve](D:\桌面\Improve_diffusion_text2video_generation_edit\improve.png)

###三、视频编辑：

####3.1 编辑功能设计

#####3.1.1基础编辑模块

尺寸调整与几何变换可以自定义尺寸与宽高比保持，用户可通过滑动条指定 GIF 宽度和高度（范围 50-1000 像素），或勾选 “保持宽高比” 自动按原始比例缩放。基于 PIL 的Image.resize实现像素重采样，采用双线性插值确保缩放后画面平滑。四方向百分比裁剪支持左、右、上、下四个方向的百分比裁剪（0-100%），计算裁剪区域坐标后调用Image.crop()去除边缘冗余内容，保持主体居中。90° 倍数旋转通过Image.rotate()实现 0°、90°、180°、270° 旋转，自动扩展画布以避免内容截断，确保帧间几何变换的一致性。

水印与视觉效果调整，其中文字水印定制，用户可输入任意文本，设置字体大小（10-50 像素）、透明度（0-255）和位置（左上、右上、左下、右下、居中）。通过ImageDraw.Draw.text()在指定位置绘制水印，利用 RGBA 通道实现半透明叠加，避免遮挡主体内容。

亮度与对比度调整是基于ImageEnhance.Brightness和ImageEnhance.Contrast模块，支持 0.1-3.0 倍的动态调整。例如，亮度值 1.0 为原始效果，0.5 为半暗，2.0 为提亮，通过像素值线性变换实现视觉增强。

#####3.1.2播放控制

通过修改 GIF 帧的持续时间（duration参数）实现播放速度控制。原始帧持续时间通常为 100 毫秒，用户可通过滑动条设置 0.1-5.0 倍速（对应持续时间 1000 毫秒至 20 毫秒），调用Image.save()时指定duration参数生成新 GIF。

#####3.2 代码实现与交互

基于ipywidgets构建交互式界面，包含文件上传、参数滑块、实时预览按钮文件上传组件中FileUpload控件支持 GIF 文件上传，自动解析二进制数据并预览原始内容。

参数调节滑块与下拉菜单包含尺寸调整：IntSlider控制宽度、高度，Checkbox控制宽高比锁定。裁剪与旋转：IntSlider设置裁剪百分比，IntSlider（步长 90）控制旋转角度。水印参数：Text输入框、IntSlider（大小 / 透明度）、Dropdown（位置选择）。播放速度：FloatSlider实现 0.1-5.0 倍速调节。操作按钮包含“预览效果” 按钮触发实时处理并显示结果；“下载 GIF” 按钮保存处理后的文件。

核心处理逻辑：

class GIFProcessor: 

  def process_gif(self): 

​    \# 帧裁剪、缩放、旋转、亮度/对比度调整、水印叠加  

​    for frame in frames: 

​      frame = frame.crop((left, top, right, bottom)) 

​      frame = frame.resize((width, height)) 

​      frame = ImageEnhance.Brightness(frame).enhance(brightness) 

​      \# 水印绘制  

​      draw.text(position, text, fill=(255,255,255,opacity)) 

​    \# 保存处理后GIF，调整播放速度

 

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps31.jpg) 

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps32.jpg) 

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps33.jpg) 

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml13324\wps34.png) 

图X编辑水印效果

###四、代码说明：

diffusion_improve.py为改进后的扩散模型代码，trainer文件夹下是不同数据集的训练参数设计，

- `train.py`是训练模型的主脚本，负责加载配置、初始化模型组件并启动训练流程
- `generate.py`用于利用训练好的模型根据文本提示生成视频内容

video_change_ui.py用于视频编辑部分，可根据需要的部分修改。

本研究在google colab上进行，数据集较大时使用A100，数据集较小时使用L4。

msrvtt_download.py为下载好的公共数据集1的文件；moving_minist_data_download.py为下载公共数据集2的文件；training_data(we_made)文件夹为我们自己爬取收集拍摄并处理好的数据集。