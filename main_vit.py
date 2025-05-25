import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    ViTForImageClassification,
    set_seed,
)

from dataset.vision import vision_dataset
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from prune.fisher import collect_mask_grads
from prune.search import search_mac, search_latency
from prune.rearrange import rearrange_mask
from prune.rescale_vit import rescale_mask_vit
from evaluate.vision import test_accuracy_vit
from utils.schedule import get_pruning_schedule


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True, choices=[
    "cifar10",
    "cifar100", 
    "timm/mini-imagenet",
    "fashion_mnist",
])
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--metric", type=str, choices=[
    "mac",
    "latency",
], default="mac")
parser.add_argument("--constraint", type=float, required=True,
    help="MAC/latency constraint relative to the original model",
)
parser.add_argument("--mha_lut", type=str, default=None)
parser.add_argument("--ffn_lut", type=str, default=None)
parser.add_argument("--num_samples", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--drop_rearrange", action="store_true",
    help="Whether to skip the rearrangement step", default=False
)
parser.add_argument("--drop_rescale", action="store_true",
    help="Whether to skip the rescaling step", default=False
)
parser.add_argument("--skip_first_eval", action="store_true",
    help="Whether to skip the first evaluation before pruning", default=False
)


def main():
    args = parser.parse_args()
    IS_LARGE = "large" in args.model_name
    img_size = 224
    # For ViT, sequence length is determined by patch size and image size
    # Default ViT-Base: 16x16 patches on 224x224 image = 196 patches + 1 [CLS] = 197
    seq_len = (img_size // 16) ** 2 + 1  # Assuming patch size of 16

    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            args.metric,
            str(args.constraint),
            f"seed_{args.seed}"
        )
    if args.drop_rearrange:
        args.output_dir += "/no_rearrange"
    elif args.drop_rescale:
        args.output_dir += "/no_rescale"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)

    # Set a GPU and the experiment seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the finetuned ViT model and the corresponding image processor
    config = AutoConfig.from_pretrained(args.ckpt_dir)
    model = ViTForImageClassification.from_pretrained(args.ckpt_dir, config=config)
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name,
        use_auth_token=None,
        use_fast=True
    )
    
    # Prepare the model
    model = model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()
    
    if not args.skip_first_eval:
        test_acc = test_accuracy_vit(model, full_head_mask, full_neuron_mask, image_processor, args.task_name)
        logger.info(f"{args.task_name} Test accuracy: {test_acc:.4f}")


    training_dataset = vision_dataset(
        args.task_name,
        image_processor=image_processor,
        training=True
    )

    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist(),
    )
    sample_batch_size = int(32 * (0.5 if IS_LARGE else 1))  # Adjust for ViT
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    start = time.time()
    # Search the optimal mask
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
    )
    teacher_constraint = get_pruning_schedule(target=args.constraint, num_iter=2)[0]
    
    if args.metric == "mac":
        teacher_head_mask, teacher_neuron_mask = search_mac(
            config,
            head_grads,
            neuron_grads,
            seq_len,
            teacher_constraint,
        )
        head_mask, neuron_mask = search_mac(
            config,
            head_grads,
            neuron_grads,
            seq_len,
            args.constraint,
        )
        pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
        logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
    elif args.metric == "latency":
        mha_lut = torch.load(args.mha_lut)
        ffn_lut = torch.load(args.ffn_lut)
        teacher_head_mask, teacher_neuron_mask = search_latency(
            config,
            head_grads,
            neuron_grads,
            teacher_constraint,
            mha_lut,
            ffn_lut,
        )
        head_mask, neuron_mask = search_latency(
            config,
            head_grads,
            neuron_grads,
            args.constraint,
            mha_lut,
            ffn_lut,
        )
        pruned_latency = estimate_latency(mha_lut, ffn_lut, head_mask, neuron_mask)
        logger.info(f"Pruned Model Latency: {pruned_latency:.2f} ms")

    # Rearrange the mask
    if not args.drop_rearrange:
        head_mask = rearrange_mask(head_mask, head_grads)
        neuron_mask = rearrange_mask(neuron_mask, neuron_grads)

    # Rescale the mask by solving a least squares problem
    if not args.drop_rescale and not args.drop_rearrange:
        head_mask, neuron_mask = rescale_mask_vit(
            model,
            config,
            teacher_head_mask,
            teacher_neuron_mask,
            head_mask,
            neuron_mask,
            sample_dataloader,
        )

    # Print the pruning time
    end = time.time()
    logger.info(f"{args.task_name} Pruning time (s): {end - start}")

    # Evaluate the accuracy
    test_acc = test_accuracy_vit(model, head_mask, neuron_mask, image_processor, args.task_name)
    logger.info(f"{args.task_name} Test accuracy: {test_acc:.4f}")

    # Save the masks
    torch.save(head_mask, os.path.join(args.output_dir, "head_mask.pt"))
    torch.save(neuron_mask, os.path.join(args.output_dir, "neuron_mask.pt"))


if __name__ == "__main__":
    main()