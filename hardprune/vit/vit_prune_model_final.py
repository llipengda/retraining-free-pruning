import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification, ViTModel
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings, ViTEncoder, ViTLayer, ViTSdpaAttention, 
    ViTSdpaSelfAttention, ViTSelfOutput, ViTIntermediate, ViTOutput
)
from safetensors import safe_open
from safetensors.torch import load_file
import math
import os

class PrunedViTSdpaSelfAttention(nn.Module):
    def __init__(self, config, keep_heads):
        super().__init__()
        self.keep_heads = keep_heads
        self.num_attention_heads = len(keep_heads)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Use SDPA (Scaled Dot Product Attention)
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        return outputs

class PrunedViTSelfOutput(nn.Module):
    def __init__(self, config, pruned_attention_dim):
        super().__init__()
        self.dense = nn.Linear(pruned_attention_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class PrunedViTSdpaAttention(nn.Module):
    def __init__(self, config, keep_heads):
        super().__init__()
        self.attention = PrunedViTSdpaSelfAttention(config, keep_heads)
        # 计算剪枝后的注意力维度
        attention_head_size = int(config.hidden_size / config.num_attention_heads)
        pruned_attention_dim = len(keep_heads) * attention_head_size
        self.output = PrunedViTSelfOutput(config, pruned_attention_dim)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class PrunedViTIntermediate(nn.Module):
    def __init__(self, config, keep_neurons):
        super().__init__()
        self.keep_neurons = keep_neurons
        intermediate_size = len(keep_neurons)
        self.dense = nn.Linear(config.hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = nn.GELU()
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class PrunedViTOutput(nn.Module):
    def __init__(self, config, keep_neurons):
        super().__init__()
        self.keep_neurons = keep_neurons
        intermediate_size = len(keep_neurons)
        self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class PrunedViTLayer(nn.Module):
    def __init__(self, config, keep_heads, keep_neurons):
        super().__init__()
        self.attention = PrunedViTSdpaAttention(config, keep_heads)
        self.intermediate = PrunedViTIntermediate(config, keep_neurons)
        self.output = PrunedViTOutput(config, keep_neurons)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # Self-attention with pre-layernorm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # Apply residual connection to attention output
        hidden_states = attention_output + hidden_states

        # Feed forward with pre-layernorm and residual connection
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

class PrunedViTEncoder(nn.Module):
    def __init__(self, config, head_mask, neuron_mask):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList()
        
        for i in range(config.num_hidden_layers):
            keep_heads = head_mask[i].nonzero(as_tuple=True)[0].tolist()
            keep_neurons = neuron_mask[i].nonzero(as_tuple=True)[0].tolist()
            layer = PrunedViTLayer(config, keep_heads, keep_neurons)
            self.layer.append(layer)

    def forward(self, hidden_states, head_mask=None, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

class PrunedViTModel(nn.Module):
    def __init__(self, config, head_mask, neuron_mask, add_pooling_layer=True):
        super().__init__()
        self.config = config

        self.embeddings = ViTEmbeddings(config)
        self.encoder = PrunedViTEncoder(config, head_mask, neuron_mask)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values=None, head_mask=None, output_attentions=None, output_hidden_states=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return (sequence_output,) + encoder_outputs[1:]

class PrunedViTForImageClassification(nn.Module):
    def __init__(self, config, head_mask, neuron_mask):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.vit = PrunedViTModel(config, head_mask, neuron_mask, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, pixel_values=None, head_mask=None, labels=None, output_attentions=None, output_hidden_states=None):
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])  # Use [CLS] token

        return logits

def load_and_transfer_weights(model, state_dict, head_mask, neuron_mask):
    """
    将原始ViT权重转移到剪枝后的模型中
    支持缩放mask：0表示剪除，非零值表示保留并缩放
    """
    
    # 获取配置信息
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    # 转移embedding权重
    embedding_keys = [k for k in state_dict.keys() if k.startswith('vit.embeddings')]
    for key in embedding_keys:
        new_key = key  # 保持原有的key结构
        if new_key in model.state_dict():
            model.state_dict()[new_key].copy_(state_dict[key])
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    
    # 确保mask在正确的设备上
    if hasattr(head_mask, 'to'):
        head_mask = head_mask.to(device)
    if hasattr(neuron_mask, 'to'):
        neuron_mask = neuron_mask.to(device)
    
    # 转移每层的权重
    for layer_idx in range(len(model.vit.encoder.layer)):
        # 获取保留的头和神经元的索引以及对应的缩放因子
        head_keep_mask = head_mask[layer_idx] != 0
        neuron_keep_mask = neuron_mask[layer_idx] != 0
        
        keep_heads = head_keep_mask.nonzero(as_tuple=True)[0].tolist()
        keep_neurons = neuron_keep_mask.nonzero(as_tuple=True)[0].tolist()
        
        # 获取缩放因子，确保在正确设备上
        head_scale_factors = head_mask[layer_idx][keep_heads].to(device)
        neuron_scale_factors = neuron_mask[layer_idx][keep_neurons].to(device)
        
        # 注意力权重转移 (QKV)
        for weight_type in ['query', 'key', 'value']:
            old_key = f'vit.encoder.layer.{layer_idx}.attention.attention.{weight_type}.weight'
            if old_key in state_dict:
                old_weight = state_dict[old_key]  # shape: [768, 768]
                # 确保权重在正确设备上
                old_weight = old_weight.to(device)
                
                # 重塑为 (num_heads, head_dim, hidden_size) 
                old_weight_reshaped = old_weight.view(num_heads, head_dim, config.hidden_size)
                # 选择保留的头
                new_weight = old_weight_reshaped[keep_heads]  # shape: [len(keep_heads), head_dim, hidden_size]
                
                # 应用缩放因子
                head_scale_factors_expanded = head_scale_factors.view(-1, 1, 1)  # [num_keep_heads, 1, 1]
                new_weight = new_weight * head_scale_factors_expanded
                
                # 重塑回 (new_attention_dim, hidden_size)
                new_weight = new_weight.view(-1, config.hidden_size)
                
                target_param = model.vit.encoder.layer[layer_idx].attention.attention.__getattr__(weight_type).weight
                
                if new_weight.shape == target_param.shape:
                    target_param.data.copy_(new_weight)
        
        # 注意力偏置转移 (QKV)
        for weight_type in ['query', 'key', 'value']:
            old_key = f'vit.encoder.layer.{layer_idx}.attention.attention.{weight_type}.bias'
            if old_key in state_dict:
                old_bias = state_dict[old_key]  # shape: [768]
                # 确保偏置在正确设备上
                old_bias = old_bias.to(device)
                # 重塑为 (num_heads, head_dim)
                old_bias_reshaped = old_bias.view(num_heads, head_dim)
                # 选择保留的头
                new_bias = old_bias_reshaped[keep_heads]  # shape: [num_keep_heads, head_dim]
                
                # 应用缩放因子
                head_scale_factors_expanded = head_scale_factors.view(-1, 1)  # [num_keep_heads, 1]
                new_bias = new_bias * head_scale_factors_expanded
                
                new_bias = new_bias.view(-1)  # 展平
                
                target_param = model.vit.encoder.layer[layer_idx].attention.attention.__getattr__(weight_type).bias
                if new_bias.shape == target_param.shape:
                    target_param.data.copy_(new_bias)
        
        # 注意力输出权重转移 - 使用自定义的PrunedViTSelfOutput
        old_key = f'vit.encoder.layer.{layer_idx}.attention.output.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]  # shape: [768, 768]
            # 重塑为 (hidden_size, num_heads, head_dim)
            old_weight_reshaped = old_weight.view(config.hidden_size, num_heads, head_dim)
            # 选择保留的头对应的列
            new_weight = old_weight_reshaped[:, keep_heads, :].contiguous()  # [hidden_size, num_keep_heads, head_dim]
            
            # 应用缩放因子 - 注意这里输出权重不需要缩放，因为输入已经被缩放了
            # 但如果想要保持一致性，可以选择性地应用缩放
            # head_scale_factors_expanded = head_scale_factors.view(1, -1, 1)  # [1, num_keep_heads, 1]
            # new_weight = new_weight * head_scale_factors_expanded
            
            new_weight = new_weight.view(config.hidden_size, -1)
            
            target_param = model.vit.encoder.layer[layer_idx].attention.output.dense.weight
            
            if new_weight.shape == target_param.shape:
                target_param.data.copy_(new_weight)
        
        # 注意力输出偏置转移 - 偏置维度不变，仍然是hidden_size
        old_key = f'vit.encoder.layer.{layer_idx}.attention.output.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            target_param = model.vit.encoder.layer[layer_idx].attention.output.dense.bias
            if old_bias.shape == target_param.shape:
                target_param.data.copy_(old_bias)
        
        # 注意力LayerNorm转移 (ViT使用Pre-LN)
        for norm_param in ['weight', 'bias']:
            old_key = f'vit.encoder.layer.{layer_idx}.layernorm_before.{norm_param}'
            if old_key in state_dict:
                old_param = state_dict[old_key]
                target_param = model.vit.encoder.layer[layer_idx].layernorm_before.__getattr__(norm_param)
                if old_param.shape == target_param.shape:
                    target_param.data.copy_(old_param)
        
        # MLP权重转移
        old_key = f'vit.encoder.layer.{layer_idx}.intermediate.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]  # shape: [intermediate_size, hidden_size]
            # 确保权重在正确设备上
            old_weight = old_weight.to(device)
            new_weight = old_weight[keep_neurons]  # 选择保留的神经元行
            
            # 应用缩放因子
            neuron_scale_factors_expanded = neuron_scale_factors.view(-1, 1)  # [num_keep_neurons, 1]
            new_weight = new_weight * neuron_scale_factors_expanded
            
            target_param = model.vit.encoder.layer[layer_idx].intermediate.dense.weight
            if new_weight.shape == target_param.shape:
                target_param.data.copy_(new_weight)
        
        old_key = f'vit.encoder.layer.{layer_idx}.intermediate.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]  # shape: [intermediate_size]
            # 确保偏置在正确设备上
            old_bias = old_bias.to(device)
            new_bias = old_bias[keep_neurons]  # 选择保留的神经元
            
            # 应用缩放因子
            new_bias = new_bias * neuron_scale_factors
            
            target_param = model.vit.encoder.layer[layer_idx].intermediate.dense.bias
            if new_bias.shape == target_param.shape:
                target_param.data.copy_(new_bias)
        
        old_key = f'vit.encoder.layer.{layer_idx}.output.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]  # shape: [hidden_size, intermediate_size]
            # 确保权重在正确设备上
            old_weight = old_weight.to(device)
            new_weight = old_weight[:, keep_neurons]  # 选择保留的神经元列
            
            # MLP输出权重不需要额外缩放，因为输入已经被缩放了
            # 但如果想要保持一致性，可以选择性地应用缩放
            # neuron_scale_factors_expanded = neuron_scale_factors.view(1, -1)  # [1, num_keep_neurons]
            # new_weight = new_weight * neuron_scale_factors_expanded
            
            target_param = model.vit.encoder.layer[layer_idx].output.dense.weight
            if new_weight.shape == target_param.shape:
                target_param.data.copy_(new_weight)
        
        old_key = f'vit.encoder.layer.{layer_idx}.output.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            target_param = model.vit.encoder.layer[layer_idx].output.dense.bias
            if old_bias.shape == target_param.shape:
                target_param.data.copy_(old_bias)
        
        # MLP LayerNorm转移 (ViT使用Pre-LN)
        for norm_param in ['weight', 'bias']:
            old_key = f'vit.encoder.layer.{layer_idx}.layernorm_after.{norm_param}'
            if old_key in state_dict:
                old_param = state_dict[old_key]
                target_param = model.vit.encoder.layer[layer_idx].layernorm_after.__getattr__(norm_param)
                if old_param.shape == target_param.shape:
                    target_param.data.copy_(old_param)
    
    # 转移最终LayerNorm权重
    for norm_param in ['weight', 'bias']:
        old_key = f'vit.layernorm.{norm_param}'
        if old_key in state_dict:
            target_param = model.vit.layernorm.__getattr__(norm_param)
            if state_dict[old_key].shape == target_param.shape:
                target_param.data.copy_(state_dict[old_key])
    
    # 转移分类器权重
    classifier_keys = ['classifier.weight', 'classifier.bias']
    for key in classifier_keys:
        if key in state_dict:
            target_param = model.classifier.__getattr__(key.split('.')[1])
            if state_dict[key].shape == target_param.shape:
                target_param.data.copy_(state_dict[key])


def load_safetensors_weights(model_path):
    """
    加载safetensors格式的权重文件
    """
    if model_path.endswith('.safetensors'):
        return load_file(model_path)
    elif model_path.endswith('.bin') or model_path.endswith('.pt'):
        return torch.load(model_path, map_location='cpu')
    else:
        # 尝试两种格式
        safetensors_path = model_path.replace('.bin', '.safetensors').replace('.pt', '.safetensors')
        if os.path.exists(safetensors_path):
            return load_file(safetensors_path)
        elif os.path.exists(model_path):
            return torch.load(model_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"找不到权重文件: {model_path} 或 {safetensors_path}")


def getPrunedViTModel():
    try:
        # 修改为ViT相关路径
        head_mask_path = "outputs/google/vit-base-patch16-224/cifar10/mac/0.5/seed_0/head_mask.pt"
        neuron_mask_path = "outputs/google/vit-base-patch16-224/cifar10/mac/0.5/seed_0/neuron_mask.pt"
        model_path = "pretrained/models/vit-base-patch16-224/cifar10/model.safetensors"  # 支持safetensors格式
        
        #用于权重剪枝和权重迁移
        head_mask = torch.load(head_mask_path)
        neuron_mask = torch.load(neuron_mask_path)
        
        #用于结构剪枝
        head_mask_bool = head_mask.bool()
        neuron_mask_bool = neuron_mask.bool()
        
        # 使用ViT配置
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=10)
        model = PrunedViTForImageClassification(config, head_mask_bool, neuron_mask_bool)
        
        state_dict = load_safetensors_weights(model_path)
        
        # 使用自定义权重转移函数
        load_and_transfer_weights(model, state_dict, head_mask,  neuron_mask)
        
        # 简单的前向传播测试
        # ViT输入是图像张量 (batch_size, channels, height, width)
        test_input = torch.randn(2, 3, 224, 224)  # batch_size=2, 3通道, 224x224图像
        try:
            with torch.no_grad():
                output = model(test_input)
        except Exception as e:
            pass
        
        return model
        
    except FileNotFoundError as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    model = getPrunedViTModel()