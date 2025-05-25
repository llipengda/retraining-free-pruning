import torch
from tqdm import tqdm

from utils.arch import apply_neuron_mask

@torch.no_grad()
def eval_vit_acc(model, head_mask, neuron_mask, dataloader, task_name):
    """
    Evaluate ViT model accuracy with given masks
    
    Args:
        model: ViT model
        head_mask: attention head mask
        neuron_mask: neuron mask for FFN layers
        dataloader: evaluation dataloader
        task_name: vision task name
    
    Returns:
        accuracy: evaluation accuracy
    """
    model.eval()
    handles = apply_neuron_mask(model, neuron_mask)
    
    progress_bar = tqdm(dataloader, desc=f"Evaluating {task_name}")
    total, correct = 0, 0
    
    for batch in progress_bar:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)
        
        # Forward pass with head mask
        outputs = model(**batch, head_mask=head_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# Wrapper function to match the original interface
@torch.no_grad() 
def test_accuracy_vit(model, head_mask, neuron_mask, image_processor, task_name):
    """
    Test accuracy wrapper function to match original interface
    This function would need a dataloader - you'd typically call it like:
    
    test_dataloader = vision_dataloader(task_name, training=False, batch_size=64)
    accuracy = test_accuracy_vit(model, head_mask, neuron_mask, image_processor, task_name, test_dataloader)
    """
    # Import here to avoid circular imports
    from dataset.vision import vision_dataloader
    
    print(f"Evaluating {task_name} with ViT model...")
    # Create test dataloader
    test_dataloader = vision_dataloader(
        task_name, 
        training=False,
        image_processor=image_processor,
        batch_size=64,
        num_workers=4
    )
    print(f"Test dataloader created with {len(test_dataloader)} batches.")
    
    # Evaluate
    accuracy = eval_vit_acc(model, head_mask, neuron_mask, test_dataloader, task_name)
    return accuracy
