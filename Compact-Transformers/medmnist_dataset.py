
from medmnist import INFO
import medmnist
from torchvision import transforms
from torch.utils.data import Dataset

class MedMNISTDataset(Dataset):
    def __init__(self, split='train', data_flag='pneumoniamnist', img_size=224):
        self.info = INFO[data_flag]
        DataClass = getattr(medmnist, self.info['python_class'])
        self.data = DataClass(split=split, download=True, size=img_size)

        self.img_size = img_size
        self.is_rgb = self.info['n_channels'] == 3

        # Transform to tensor only, since image is already 224x224
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # Ensure grayscale images are converted to RGB, robust to both numpy arrays and PIL Images
        from PIL import Image
        if isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            img = Image.fromarray(img.astype('uint8'))
            if img.mode != 'RGB':
                img = img.convert('RGB')

        img = self.transform(img)
        import torch
        label = torch.FloatTensor(label) if isinstance(label, list) or isinstance(label, tuple) else torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return len(self.info['label'])

    def get_num_channels(self):
        return self.info['n_channels']

    def get_class_names(self):
        return self.info['label']

################ uncomment if you want to visualize the dataset ################
import os
import matplotlib.pyplot as plt
from PIL import Image
from medmnist import INFO


# Datasets and their custom config
DATASETS = [
    "bloodmnist",
    "chestmnist",
    "octmnist",
    "pneumoniamnist",
    "breastmnist",
    "dermamnist",
    "pathmnist",
    "tissuemnist",
    "retinamnist",
    "organamnist",
    "organsmnist",
    "organmcnist"
]

# Parameters
N_IMAGES_PER_DATASET = 16
IMG_SIZE = 224

fig, axes = plt.subplots(
    nrows=2,
    ncols=6,
    figsize=(24, 8)
)

for ax, data_flag in zip(axes.flatten(), DATASETS):
    dataset = MedMNISTDataset(split='train', data_flag=data_flag, img_size=IMG_SIZE)

    n_images = min(N_IMAGES_PER_DATASET, len(dataset))
    n_cols = int(n_images ** 0.5)
    n_rows = (n_images + n_cols - 1) // n_cols

    combined = Image.new('RGB', (n_cols * IMG_SIZE, n_rows * IMG_SIZE))

    for idx in range(n_images):
        img, _ = dataset[idx]
        # img is a torch tensor [3, H, W] — convert to PIL
        pil_img = transforms.ToPILImage()(img)
        row = idx // n_cols
        col = idx % n_cols
        combined.paste(pil_img, (col * IMG_SIZE, row * IMG_SIZE))

    ax.imshow(combined)
    ax.set_title(data_flag, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('medmnist_overview.png', dpi=300, bbox_inches='tight')
plt.show()