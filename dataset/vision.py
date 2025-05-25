import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datasets import load_dataset

label_key_map = {
    "cifar10": "label",
    "cifar100": "fine_label",
    "timm/mini-imagenet": "label",
    "fashion_mnist": "label"
}

img_key_map = {
    "cifar10": "img",
    "cifar100": "img",
    "timm/mini-imagenet": "image",
    "fashion_mnist": "image"
}

class VisionDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, task_name):
        self.dataset = hf_dataset
        self.label_key = label_key_map.get(task_name, "label")
        self.img_key = img_key_map.get(task_name, "img")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.img_key]
        label = item[self.label_key]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        image = self.transform(image)
        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long)
        }

def vision_dataset(task_name, image_processor, training=True):
    split = "train" if training else "test"
    if task_name == "timm/mini-imagenet":
        split = "train" if training else "validation"
    dataset = load_dataset(task_name, split=split)
    return VisionDataset(dataset, image_processor, task_name)

def vision_dataloader(task_name, image_processor, training=True, batch_size=32, num_workers=4):
    dataset = vision_dataset(task_name, image_processor, training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True
    )
