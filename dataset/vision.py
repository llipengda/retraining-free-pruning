import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datasets import load_dataset

label_key_map = {
    "cifar10": "label",
    "cifar100": "fine_label",
}

class VisionDataset(Dataset):
    def __init__(self, hf_dataset, image_processor, task_name):
        self.dataset = hf_dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])
        self.transform = transform
        self.label = label_key_map.get(task_name, "label")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label = item[self.label]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transform(image)
        return {
            "pixel_values": image,
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
def vision_dataset(task_name, image_processor, training=True):
    if training:
        return VisionDataset(load_dataset(task_name, split="train"), image_processor, task_name)
    else:
        return VisionDataset(load_dataset(task_name, split="test"), image_processor, task_name)
    
def vision_dataloader(task_name, image_processor, training=True, batch_size=32, num_workers=4):
    dataset = vision_dataset(task_name, image_processor, training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=True
    )