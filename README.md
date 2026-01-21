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
```
