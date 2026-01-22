## AugCD 主要方法说明

- `__init__`：模型初始化，组装各个模块（backbone、text_encoder、context_decoder、decode_head等）。
- `_init_decode_head`：初始化 decode_head（分割头），用于语义分割预测。
- `_init_auxiliary_head`：初始化辅助分割头（可选，用于深监督）。
- `_init_identity_head`：初始化身份分割头（可选，用于额外监督）。
- `extract_feat(inputs)`：提取输入图像的多层特征，调用 backbone（如 CLIPResNetWithAttention）。
- `encode_decode(inputs, batch_img_metas)`：编码输入图像并解码为分割结果。包括特征提取、文本特征融合、上下文解码、特征融合、分割预测等。
- `after_extract_feat_clip(x, text)`：视觉特征与文本特征进行上下文融合，得到最终的融合特征。
- `get_cls_text(img_infos, train=True)`：生成每张图片的文本描述（如类别、变化区域等），用于文本特征编码。
- `loss(inputs, data_samples)`：计算模型损失，包括分割损失、文本融合损失等。训练时的主入口。
- `_decode_head_forward_train(inputs, data_samples)`：调用 decode_head 的 loss 方法，计算主分割损失。
- `_decode_head_forward_train_with_text(x, fuse_diff, textA, textB, data_samples)`：调用 decode_head 的 custom_loss 方法，计算融合文本的分割损失。
- `_auxiliary_head_forward_train(inputs, data_samples)`：计算辅助分割头的损失（如有）。
- `_identity_head_forward_train(x, data_samples, loss_id)`：计算身份分割头的损失（如有）。
- `predict(inputs, data_samples=None)`：推理接口，输入图像，输出分割结果（包括像素类别和分割概率）。
- `_forward(inputs, data_samples=None)`：网络前向过程，主要用于模型导出、ONNX等场景。
- `mm_slide_inference(inputs, batch_img_metas)`：多模态滑窗推理，适用于大图分块推理。
- `slide_inference(inputs, batch_img_metas)`：普通滑窗推理，适用于大图分块推理。
- `whole_inference(inputs, batch_img_metas)`：整图推理，直接对整张图片进行分割预测。
- `inference(inputs, batch_img_metas)`：推理主入口，根据配置选择滑窗或整图推理。

## 使用指南

### 数据准备

本工作使用两个遥感变化检测（RSCD）数据集：LEVIR Lab 的 [LEVIR-CD](https://justchenhao.github.io/LEVIR/) 与 Q. Shi 等人的 [SYSU-CD](https://github.com/liumency/SYSU-CD)。

1. 下载两个数据集，并按如下结构重新组织：

  ```
  Dataset/LEVIRCD/train/time1/
  Dataset/LEVIRCD/train/time2/
  Dataset/LEVIRCD/train/label/
  Dataset/LEVIRCD/val/...
  Dataset/LEVIRCD/test/...
  Dataset/SYSUCD/...
  ```

2. 运行 [`tools/write_path.py`](./tools/write_path.py) 生成三个文本文件：`Dataset/LEVIRCD/train.txt`、`Dataset/LEVIRCD/val.txt` 和 `Dataset/LEVIRCD/test.txt`。然后修改该脚本，为 SYSU-CD 数据集同样生成对应的三个文件。

3. 从 [Hugging Face](https://huggingface.co/YarnYang/MdaCD) 下载 MdaCD 官方提供的两个数据集的 CLIP 文件（也可以后续自己生成 CLIP 文件），并按如下结构导入与整理：

  ```
  Dataset/clip_files/LEVIRCD/train/*.json
  Dataset/clip_files/LEVIRCD/val/*.json
  Dataset/clip_files/LEVIRCD/test/*.json
  Dataset/clip_files/SYSUCD/...
  ```

你也可以使用自定义数据集，只需按照上述步骤组织即可。若有不清楚的细节，可参考 [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP) 的数据准备指南。

### 环境配置

本工作使用与 [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP) 相同的环境配置。若有不清楚的细节，可参考其环境配置指南。

1. 环境要求

  ```
  Ubuntu 20.04 (Focal)
  CUDA 12.8
  一块至少 16GB 显存的 NVIDIA GPU
  ```

2. 创建 Python 3.8 环境，并安装必要依赖：

  ```
  torch==2.0.0
  torchvision==0.15.1
  numpy==1.24.3
  ```

  其余依赖可以在训练过程中再安装。

3. 安装 CLIP（用于生成 CLIP 文件）。执行：

  ```
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  ```

  注意：使用其他方式安装 CLIP 也可以。

### 训练与测试命令

1. （可选）生成 CLIP 文件。执行：

  ```
  cd Masking
  python masking_LEVIRCD.py
  python masking_SYSUCD.py
  ```

  上述脚本会生成带 mask 的图像。注意：该脚本可能会在 `./Masking` 目录下写入临时文件。

  然后使用生成的 masked 图像来生成 CLIP 文件：

  ```
  cd tools
  bash clip.sh

cd tools
python general/clip_inference.py \
  --src_path "../Dataset/LEVIRCD" \
  --split train val test \
  --img_split penalty_10_mask1 penalty_10_mask2 \
  --class_names_path "../Masking/rscls.txt" \
  --model_name "ViT-B/16" \
  --tag "56_vit16"

 使用起始提供的需要这个，否则注释
  python copy_clip_files.py LEVIRCD penalty_10_mask
  python copy_clip_files.py SYSUCD penalty_100_mask
生成完mask的json之后，修改tools下面的copy_files
  source_file = f"/home/dell/gitrepos/MdaCD/Dataset/{dataset}/{folder}/{mask}{i}_clipcls_56_vit16.json"
  ```

2. 训练 MdaCD：

  ```
  bash tr.sh
  ```

3. 测试 MdaCD：

  ```
  bash te.sh
  ```

## 许可证（License）

本仓库基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 构建，其遵循 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)。

## 致谢（Acknowledgement）

本工作基于 OpenMMLab 的 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 与 S. Dong 等人的 [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)。感谢他们的优秀工作！

## 修改的部分

```
Loops当中的FP和FN指标进行交换了两次
loss当中的mask进行修改
pkill -9 -f "python.*train.py"
```
