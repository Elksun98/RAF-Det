# RAF-Det

**Response-Aware Feature Alignment and Fourier-Domain Reparameterization for Unsupervised Day-to-Night Domain-Adaptive Object Detection**

RAF-Det 是一个面向**无监督昼夜（白天→黑夜）域自适应目标检测**的开源实现。模型以 YOLOv7 双主干结构为基础，结合**响应感知特征对齐（Response-Aware Feature Alignment）**与**傅里叶域重参数化（Fourier-Domain Reparameterization）**，在仅使用源域（白天 / 正常天气）标注、目标域（黑夜 / 雾天）无标注的条件下，提升模型在目标域上的检测性能。

---
<img width="1136" height="427" alt="270c219069c2fcc4e150d22e96be5723" src="https://github.com/user-attachments/assets/7353132b-e188-4d1f-b577-e11bcf37a83c" />
<img width="1127" height="418" alt="d565fda3-e11f-48a4-a47a-7c693a68724b" src="https://github.com/user-attachments/assets/060d0544-e782-4943-b891-1278a164eb3b" />


## ✨ 方法概述

RAF-Det 在训练阶段同时输入**有标注的源域图像**与**无标注的目标域图像**，整体由以下模块组成：

- **YOLOv7 双主干（B1 + B2）+ 融合检测头**：通过跨分支特征融合（`CBLinear` / `CBFuse`）增强跨域特征表达，使用单一融合检测头（`single_fused_head`）输出多尺度预测。
- **傅里叶域重参数化（FourierBoostResidualConv）**：在主干与检测头的关键层引入动态门控高频滤波分支，对低照度 / 雾天场景中的高频边缘与纹理进行增强与去噪，并通过门控权重控制傅里叶分支的贡献。
- **响应感知特征对齐**：基于特征响应强度构建前景加权图（`build_response_weight`），并以**前景加权余弦相似性损失**约束源域与目标域特征的一致性。
- **对抗式域分类器**：在 B1（P3/P4）及 B2 弱分支上通过梯度反转层（GRL）进行域对齐，学习域不变特征。
- **动态损失平衡（可选）**：根据检测 / 域 / 相似性损失的统计量自适应调整各损失权重。

> 训练损失：`L = L_det + λ_dom · L_dom + λ_sim · L_sim + λ_dom_b2 · L_dom_b2 + λ_gate · L_gate`

---

## 📦 安装

```bash
# 1. 克隆仓库
git clone https://github.com/<your-name>/RAF-Det.git
cd RAF-Det

# 2. 创建环境（推荐 Python 3.8）
conda create -n raf-det python=3.8 -y
conda activate raf-det

# 3. 安装 PyTorch（与论文一致：CUDA 11.3 + PyTorch 1.12.1）
conda install pytorch=1.12.1 torchvision cudatoolkit=11.3 -c pytorch -y

# 4. 安装其余依赖
pip install -r requirements.txt
```

**论文实验环境**：NVIDIA RTX A100 40G · Ubuntu 18.04 · CUDA 11.3 · Python 3.8 · PyTorch 1.12.1。

---

## 🗂️ 数据集

RAF-Det 在三个昼夜 / 天气域自适应基准上进行评估：

| 数据集                 | 源域（有标注） | 目标域（无标注） | 类别数 | 数据配置                         |
| :--------------------- | :------------- | :--------------- | :----: | :------------------------------- |
| **BDD100K**            | 白天           | 黑夜             |   10   | `data/coco_bdd_uda.yaml`         |
| **SHIFT**              | 白天           | 黑夜             |   6    | `data/shift_uda.yaml`            |
| **Cityscapes → Foggy** | 正常天气       | 雾天             |   8    | `data/cityscapes_foggy_uda.yaml` |

每个数据配置遵循统一的 UDA 字段格式：

```yaml
train:  <源域训练集 .txt，含标注>      # 计算检测损失 + 域/相似性损失
val:    <源域验证集 .txt>             # 训练期间用于模型选择
train2: <目标域训练集 .txt，无标注>    # 仅参与域/相似性对齐
val2:   <目标域验证集 .txt>           # 仅用于训练结束后的最终评估
nc:     <类别数>
names:  [ ... ]
```

### 数据准备脚本

```bash
# BDD100K：按 timeofday 划分白天/黑夜，并转换为 YOLO 标签
python prepare_bdd_for_raf_det.py --bdd-root <BDD100K_ROOT> --out-root <OUT_ROOT> --link-mode symlink

# Cityscapes → Foggy Cityscapes
python prepare_cityscapes_for_raf_det.py --cityscapes-root <CITYSCAPES_ROOT> --output-root <OUT_ROOT> --fog-beta 0.02

# SHIFT（支持已整理的 COCO 布局或原始 SHIFT 布局）
python prepare_shift_for_raf_det.py --shift-root <SHIFT_ROOT>
```

> 准备完成后，请检查对应的 `data/*.yaml` 中 `train/val/train2/val2` 路径是否指向你的本地数据。

---

## 🚀 训练

所有训练均通过 `train.py` 启动，源域与目标域图像在每个 batch 内成对输入。`--batch-size` 为所有 GPU 的总批大小，源域与目标域各占一半。

### 单卡训练（BDD100K：白天 → 黑夜）

```bash
python train.py \
  --cfg cfg/dy-yolov7-uda-fusedhead-fourier-boost.yaml \
  --data data/coco_bdd_uda.yaml \
  --hyp hyp/hyp.scratch.p5.yaml \
  --weight weights/dy-yolov7.pt \
  --epochs 300 \
  --batch-size 16 \
  --img-size 800 800 \
  --device 0 \
  --workers 8 \
  --project runs/uda \
  --name bdd_uda \
  --lambda-dom 0.05 \
  --lambda-sim 0.05 \
  --lambda-dom-b2 0.01 \
  --lambda-fourier-gate 1e-4 \
  --eval-interval 5 \
  --eval-target-final
```

### 多卡分布式训练（Cityscapes → Foggy，DDP）

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 train.py \
  --cfg cfg/dy-yolov7-uda-fusedhead-fourier-boost.yaml \
  --weight weights/dy-yolov7.pt \
  --data data/cityscapes_foggy_uda.yaml \
  --epochs 300 \
  --batch-size 16 \
  --device 0,1 \
  --project runs/uda \
  --name cityscapes_foggy_uda \
  --lambda-dom 0.05 \
  --lambda-sim 0.05 \
  --eval-target-final \
  --data-fraction 1 \
  --eval-interval 5
```

### SHIFT（白天 → 黑夜）

```bash
python train.py \
  --cfg cfg/dy-yolov7-uda-fusedhead-fourier-boost.yaml \
  --data data/shift_uda.yaml \
  --hyp hyp/hyp.scratch.p5.yaml \
  --weight weights/dy-yolov7.pt \
  --epochs 300 \
  --batch-size 16 \
  --img-size 800 800 \
  --device 0 \
  --project runs/uda \
  --name shift_uda \
  --lambda-dom 0.05 \
  --lambda-sim 0.05 \
  --lambda-dom-b2 0.01 \
  --lambda-fourier-gate 1e-4 \
  --eval-interval 5 \
  --eval-target-final
```

### 常用训练参数

| 参数                        | 说明                                   | 默认                     |
| :-------------------------- | :------------------------------------- | :----------------------- |
| `--cfg`                     | 模型结构配置                           | —                        |
| `--data`                    | 数据集 UDA 配置                        | `data/coco_bdd_uda.yaml` |
| `--weight`                  | 初始化权重（YOLOv7 预训练）            | `''`                     |
| `--epochs` / `--batch-size` | 训练轮次 / 总批大小                    | `300` / `2`              |
| `--img-size`                | 训练/测试输入尺寸                      | `800 800`                |
| `--lambda-dom`              | B1 域分类损失权重                      | `0.1`                    |
| `--lambda-dom-b2`           | B2 弱域对齐权重                        | `0.01`                   |
| `--lambda-sim`              | 前景加权余弦相似性损失权重             | `0.1`                    |
| `--lambda-fourier-gate`     | 傅里叶门控正则权重                     | `1e-4`                   |
| `--freeze-bn-after`         | 指定 epoch 后冻结 BN 统计量            | `999`                    |
| `--dynamic-balance`         | 启用动态损失平衡                       | 关闭                     |
| `--eval-interval`           | 每隔 N 个 epoch 评估一次               | `5`                      |
| `--eval-target-final`       | 训练结束后在目标域 `val2` 上做最终评估 | 关闭                     |
| `--data-fraction`           | 使用数据列表的比例                     | `1.0`                    |

---

## 🔬 消融实验

`run_ablation_study.py` 会为每个消融变体**独立重新训练**一个模型，保证对比公平。共 5 个变体：

| 变体            | 含义               | 实现方式                                 |
| :-------------- | :----------------- | :--------------------------------------- |
| `full`          | 完整模型           | base cfg                                 |
| `no_fourier`    | 去除傅里叶重参数化 | `cfg/ablation/fusedhead-no-fourier.yaml` |
| `no_similarity` | 去除余弦相似性损失 | `--lambda-sim 0`                         |
| `no_domain`     | 去除域分类器损失   | `--lambda-dom 0 --lambda-dom-b2 0`       |
| `no_b2`         | 去除 B2 辅助分支   | `cfg/ablation/fusedhead-b1-only.yaml`    |

### 一次性运行全部 5 组

```bash
python run_ablation_study.py \
  --data data/coco_bdd_uda.yaml \
  --base-cfg cfg/dy-yolov7-uda-fusedhead-fourier-boost.yaml \
  --no-fourier-cfg cfg/ablation/fusedhead-no-fourier.yaml \
  --no-b2-cfg cfg/ablation/fusedhead-b1-only.yaml \
  --init-weight weights/dy-yolov7.pt \
  --epochs 300 \
  --batch-size 16 \
  --img-size 800 \
  --lambda-dom 0.05 \
  --lambda-sim 0.05 \
  --lambda-dom-b2 0.01 \
  --lambda-fourier-gate 1e-4 \
  --eval-interval 5 \
  --eval-target-final \
  --project runs/ablation_train \
  --name-prefix ablation
```

### 只运行指定变体

```bash
# 仅运行 no_fourier 与 no_b2
python run_ablation_study.py --variants no_fourier no_b2 \
  --data data/coco_bdd_uda.yaml \
  --init-weight weights/dy-yolov7.pt \
  --eval-target-final

# 多卡运行某一变体（DDP）
python run_ablation_study.py --variants full \
  --nproc-per-node 2 --cuda-visible-devices 2,3 \
  --data data/cityscapes_foggy_uda.yaml \
  --init-weight weights/dy-yolov7.pt \
  --eval-target-final
```

### 仅生成命令而不训练（dry-run）

```bash
python run_ablation_study.py --dry-run --data data/coco_bdd_uda.yaml
```

> 生成的命令与清单会保存到 `runs/ablation_train/<name-prefix>/`（`ablation_train_commands.sh` 与 `ablation_train_manifest.json`）。

---

## 📈 测试与评估

在目标域验证集（`val2`）上评估训练得到的模型：

```bash
python test.py \
  --cfg cfg/dy-yolov7-uda-fusedhead-fourier-boost.yaml \
  --weight runs/uda/bdd_uda/weights/best.pt \
  --data data/coco_bdd_uda.yaml \
  --img-size 800 \
  --batch-size 8 \
  --conf-thres 0.001 \
  --iou-thres 0.65 \
  --task val \
  --use-val2 \
  --device 0 \
  --verbose
```

- `--use-val2`：使用配置中的目标域验证集 `val2` 进行评估；不加该参数则评估源域 `val`。
- `--infer-source <图像文件夹>`：在评估之后对指定文件夹的图像做可视化推理。

---

## 🎨 可视化

```bash
# 傅里叶重参数化模块分析（门控/混合比例、傅里叶核与目标域检测对比等）
python visualize_fourier_reparam.py \
  --cfg cfg/dy-yolov7-uda-fusedhead-fourier-boost.yaml \
  --data data/coco_bdd_uda.yaml \
  --weight runs/uda/bdd_uda/weights/best.pt

# 绘制训练过程中的各损失曲线
python plot_training_results.py
```

---

## 📁 目录结构

```
RAF-Det/
├── cfg/                                  # 模型结构配置
│   ├── dy-yolov7-uda-fusedhead-fourier-boost.yaml   # 完整模型（默认）
│   ├── dy-yolov7-uda-fourier.yaml
│   ├── dy-yolov7-uda.yaml
│   └── ablation/
│       ├── fusedhead-no-fourier.yaml     # 消融：去傅里叶
│       └── fusedhead-b1-only.yaml        # 消融：去 B2 分支
├── data/                                 # 数据集 UDA 配置
│   ├── coco_bdd_uda.yaml                 # BDD100K
│   ├── shift_uda.yaml                    # SHIFT
│   └── cityscapes_foggy_uda.yaml         # Cityscapes → Foggy
├── hyp/hyp.scratch.p5.yaml               # 超参数
├── models/                               # 网络与傅里叶/UDA 模块
├── utils/                                # 数据、损失、训练工具
├── train.py                              # UDA 训练入口
├── test.py                               # 评估 / 推理
├── run_ablation_study.py                 # 消融实验启动器
├── prepare_bdd_for_raf_det.py            # BDD100K 数据准备
├── prepare_cityscapes_for_raf_det.py     # Cityscapes 数据准备
├── prepare_shift_for_raf_det.py          # SHIFT 数据准备
├── visualize_fourier_reparam.py          # 傅里叶模块可视化
└── plot_training_results.py              # 训练曲线绘制
```

---

## 📝 引用

如果本项目对你的研究有帮助，请考虑引用：

```bibtex
@article{rafdet2026,
  title   = {Response-Aware Feature Alignment and Fourier-Domain Reparameterization
             for Unsupervised Day-to-Night Domain-Adaptive Object Detection},
  author  = {<Your Name> and others},
  year    = {2026}
}
```

## 🙏 致谢

本项目基于 [YOLOv7](https://github.com/WongKinYiu/yolov7) 的优秀工作构建，特此致谢。

## License

本项目仅供学术研究使用。
