import torch
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torchvision import transforms
from PIL import Image
import argparse

label_key_map = {
    "cifar10": "label",
    "cifar100": "fine_label",
    "fashion_mnist": "label",
    "timm/mini-imagenet": "label",
}

eval_key_map = {
    "cifar10": "test",
    "cifar100": "test",
    "fashion_mnist": "test",
    "timm/mini-imagenet": "validation",
}

img_key_map = {
    "cifar10": "img",
    "cifar100": "img",
    "fashion_mnist": "image",
    "timm/mini-imagenet": "image",
}


class ViTTrainer:
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = None

    def prepare_dataset(self, dataset_name="cifar10", data_dir=None):
        self.dataset_name = dataset_name
        self.label_key = label_key_map.get(dataset_name, "label")
        eval_key = eval_key_map.get(dataset_name, "test")

        if dataset_name in label_key_map:
            dataset = load_dataset(dataset_name, cache_dir=data_dir)
            train_dataset = dataset["train"]
            eval_dataset = dataset[eval_key]

            labels = train_dataset.features[self.label_key].names
            self.id2label = {i: label for i, label in enumerate(labels)}
            self.label2id = {label: i for i, label in enumerate(labels)}

        elif data_dir:
            dataset = load_dataset("imagefolder", data_dir=data_dir)
            train_dataset = dataset["train"]
            if "test" in dataset:
                eval_dataset = dataset["test"]
            else:
                split = train_dataset.train_test_split(test_size=0.2)
                train_dataset = split["train"]
                eval_dataset = split["test"]

            unique_labels = sorted(set(example["label"] for example in train_dataset))
            self.id2label = {i: str(i) for i in unique_labels}
            self.label2id = {v: k for k, v in self.id2label.items()}

        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        if self.num_classes is None:
            self.num_classes = len(self.id2label)

        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id

        print(f"模型加载成功: {self.model_name}")
        print(f"类别数: {self.num_classes}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        return train_dataset, eval_dataset

    def transform_images(self, examples):
        img_key = img_key_map.get(self.dataset_name, "img")
        images = [Image.fromarray(np.array(img)).convert('RGB') for img in examples[img_key]]
        inputs = self.image_processor(images, return_tensors="pt")
        inputs["labels"] = examples[self.label_key]
        return inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    def train(self, dataset_name="cifar10", data_dir=None, output_dir="./vit-output", num_epochs=3, batch_size=32, learning_rate=5e-5):
        train_dataset, eval_dataset = self.prepare_dataset(dataset_name, data_dir)
        train_dataset.set_transform(self.transform_images)
        eval_dataset.set_transform(self.transform_images)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            warmup_steps=100,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            fp16=True,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        print("开始训练...")
        trainer.train()
        trainer.save_model()
        self.image_processor.save_pretrained(output_dir)
        print(f"模型已保存到: {output_dir}")
        return trainer

    def evaluate_model(self, model_path, test_images=None):
        model = AutoModelForImageClassification.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model.eval()

        if test_images is None:
            _, eval_dataset = self.prepare_dataset(self.dataset_name)
            img_key = img_key_map.get(self.dataset_name, "img")
            test_images = [eval_dataset[i][img_key] for i in range(5)]

        predictions = []
        with torch.no_grad():
            for img in test_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                inputs = image_processor(img, return_tensors="pt")
                outputs = model(**inputs)
                pred_idx = outputs.logits.argmax(-1).item()
                predictions.append(pred_idx)

        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224", help="Hugging Face 模型名")
    parser.add_argument("--dataset", type=str, default="cifar10", help="数据集名称：cifar10/cifar100/fashion_mnist/timm/mini-imagenet/custom")
    parser.add_argument("--data_dir", type=str, default=None, help="自定义数据集路径（imagefolder）")
    parser.add_argument("--num_classes", type=int, default=None, help="分类数（若不指定，将自动推断）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每批大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./models/{args.model.split('/')[-1]}/{args.dataset.replace('/', '_')}"

    print(f"使用模型: {args.model}，数据集: {args.dataset}")
    trainer = ViTTrainer(model_name=args.model, num_classes=args.num_classes)
    trainer.train(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    predictions = trainer.evaluate_model(args.output_dir)
    print("预测结果:", predictions)


if __name__ == "__main__":
    main()
