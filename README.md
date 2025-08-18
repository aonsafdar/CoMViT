# CoMViT

**CoMViT** (Compact Medical Vision Transformer) is a lightweight and efficient Vision Transformer designed for low-resource medical imaging tasks. It is size-configured for optimization and replaces rigid patch tokenization with optimized convolutional tokenizers, incorporates learnable sequence pooling, and applies locality-aware self-attention through diagonal masking and learnable temperature scaling techniques.

---

## 🔬 Highlights

- 🧠 Optimized for **medical image classification** under resource constraints
- ⚙️ Uses **convolutional tokenization** instead of patch splitting
- 🧩 Sequence pooling instead of CLS token
- 🔍 Locality Self-Attention (LSA) for focused attention
- 📦 ~4.5M parameters — smaller than most ViTs and CNNs

---

## 📁 Directory Structure
```bash
Compact-Transformers/
├── configs/                 # Dataset-specific YAML configs
├── src/                    # Core model code (backbones, pooling, tokenizer)
├── train.py                # Main training script
├── main_medmnist.py        # MedMNIST-specific training entry point
├── solver.py               # Loss and optimizer configuration
├── dist_train.sh           # Slurm-based distributed training launcher
├── get_flops.py            # FLOPs and parameter calculation
├── Variants.md             # Description of tested model variants
├── output/                 # Saved checkpoints and logs
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/aonsafdar/CoMViT.git
cd CoMViT/Compact-Transformers
```

### 2. Install Dependencies
We recommend using conda:
```bash
conda create -n comvit python=3.9
conda activate comvit
pip install -r requirements.txt
```

You may also need:
```bash
pip install timm medmnist torch torchvision
```

---

## 🏋️‍♂️ Training

To train `cct_7_7x2_224` on a MedMNIST dataset (e.g., PathMNIST):

```bash
python train.py \
  --model cct_7_7x2_224 \
  --dataset_name PathMNIST \
  --dataset_path ./datasets/medmnist \
  --img_size 224 \
  --batch_size 128 \
  --epochs 100 \
  --lr 5e-4 \
  --output_dir ./output
```

For training all datasets:
```bash
python main_medmnist.py --model cct_7_7x2_224
```

---

## 🧪 Evaluation
To evaluate a saved checkpoint:
```bash
python train.py \
  --eval \
  --resume ./output/best_checkpoint.pth \
  --dataset_name PathMNIST
```

---

## 🧰 Configuration Files
You can pass a YAML file for dataset-specific settings. Example:
```yaml
# configs/pathmnist.yaml
dataset: pathmnist
num_classes: 9
img_size: 224
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
batch_size: 128
lr: 5e-4
model: cct_7_7x2_224
```
Then run:
```bash
python train.py --config configs/pathmnist.yaml
```

---

## 📊 Citation
If you use CoMViT in your research, please cite:
```bibtex
@inproceedings{comvit2025,
  title     = {CoMViT: Efficient Vision Backbone for Representation Learning in Medical Imaging},
  author    = {xyz},
  booktitle = {MICCAI MIRASOL Workshop},
  year      = {2025}
}
```

---

## 🔗 Related Projects
- [CCT (Official)](https://github.com/SHI-Labs/Compact-Transformers)
- [MedMNIST](https://medmnist.com)

---

## 📬 Contact
For questions or contributions, please contact xyz

---

MIT License. See [LICENSE](../LICENSE).
