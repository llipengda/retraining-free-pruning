import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertModel

class CustomBertLayer(nn.Module):
    def __init__(self, config, layer_id, keep_heads, keep_neurons):
        super().__init__()
        self.keep_heads = keep_heads
        self.keep_neurons = keep_neurons

        self.num_heads = len(keep_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dim = self.num_heads * self.head_dim

        # Attention
        self.query = nn.Linear(config.hidden_size, self.attention_dim)
        self.key = nn.Linear(config.hidden_size, self.attention_dim)
        self.value = nn.Linear(config.hidden_size, self.attention_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.att_out = nn.Linear(self.attention_dim, config.hidden_size)
        self.att_norm = nn.LayerNorm(config.hidden_size)

        # FFN
        self.intermediate = nn.Linear(config.hidden_size, len(keep_neurons))
        self.output = nn.Linear(len(keep_neurons), config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        B, T, _ = Q.shape
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_probs @ V
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.dropout(self.att_out(attn_output))
        x = self.att_norm(x + attn_output)

        h = self.activation(self.intermediate(x))
        h = self.output(h)
        x = self.output_norm(x + h)
        return x


class PrunedBertForSequenceClassification(nn.Module):
    def __init__(self, config, head_mask, neuron_mask):
        super().__init__()
        self.config = config
        self.head_mask = head_mask
        self.neuron_mask = neuron_mask

        bert_model = BertModel(config)
        self.embeddings = bert_model.embeddings

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            keep_heads = head_mask[i].nonzero(as_tuple=True)[0].tolist()
            keep_neurons = neuron_mask[i].nonzero(as_tuple=True)[0].tolist()
            layer = CustomBertLayer(config, i, keep_heads, keep_neurons)
            self.layers.append(layer)

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_act = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        if hasattr(embedding_output, 'last_hidden_state'):
            x = embedding_output.last_hidden_state
        else:
            x = embedding_output

        for layer in self.layers:
            x = layer(x)

        pooled = self.pooler_act(self.pooler(x[:, 0]))
        logits = self.classifier(pooled)
        return logits


def load_and_transfer_weights(model, state_dict, head_mask, neuron_mask):
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    embedding_keys = [k for k in state_dict.keys() if k.startswith('bert.embeddings')]
    for key in embedding_keys:
        new_key = key.replace('bert.embeddings.', 'embeddings.')
        if new_key in model.state_dict():
            model.state_dict()[new_key].copy_(state_dict[key])

    for layer_idx in range(len(model.layers)):
        keep_heads = head_mask[layer_idx].nonzero(as_tuple=True)[0].tolist()
        keep_neurons = neuron_mask[layer_idx].nonzero(as_tuple=True)[0].tolist()

        for weight_type in ['query', 'key', 'value']:
            old_key = f'bert.encoder.layer.{layer_idx}.attention.self.{weight_type}.weight'
            if old_key in state_dict:
                old_weight = state_dict[old_key]
                old_weight_reshaped = old_weight.view(num_heads, head_dim, config.hidden_size)
                new_weight = old_weight_reshaped[keep_heads].view(-1, config.hidden_size)
                if new_weight.shape == model.layers[layer_idx].__getattr__(weight_type).weight.shape:
                    model.layers[layer_idx].__getattr__(weight_type).weight.data.copy_(new_weight)

        for weight_type in ['query', 'key', 'value']:
            old_key = f'bert.encoder.layer.{layer_idx}.attention.self.{weight_type}.bias'
            if old_key in state_dict:
                old_bias = state_dict[old_key]
                old_bias_reshaped = old_bias.view(num_heads, head_dim)
                new_bias = old_bias_reshaped[keep_heads].view(-1)
                if new_bias.shape == model.layers[layer_idx].__getattr__(weight_type).bias.shape:
                    model.layers[layer_idx].__getattr__(weight_type).bias.data.copy_(new_bias)

        old_key = f'bert.encoder.layer.{layer_idx}.attention.output.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]
            old_weight_reshaped = old_weight.view(config.hidden_size, num_heads, head_dim)
            new_weight = old_weight_reshaped[:, keep_heads, :].contiguous().view(config.hidden_size, -1)
            if new_weight.shape == model.layers[layer_idx].att_out.weight.shape:
                model.layers[layer_idx].att_out.weight.data.copy_(new_weight)

        old_key = f'bert.encoder.layer.{layer_idx}.attention.output.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            if old_bias.shape == model.layers[layer_idx].att_out.bias.shape:
                model.layers[layer_idx].att_out.bias.data.copy_(old_bias)

        for norm_param in ['weight', 'bias']:
            old_key = f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.{norm_param}'
            if old_key in state_dict:
                old_param = state_dict[old_key]
                if old_param.shape == model.layers[layer_idx].att_norm.__getattr__(norm_param).shape:
                    model.layers[layer_idx].att_norm.__getattr__(norm_param).data.copy_(old_param)

        old_key = f'bert.encoder.layer.{layer_idx}.intermediate.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]
            new_weight = old_weight[keep_neurons]
            if new_weight.shape == model.layers[layer_idx].intermediate.weight.shape:
                model.layers[layer_idx].intermediate.weight.data.copy_(new_weight)

        old_key = f'bert.encoder.layer.{layer_idx}.intermediate.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            new_bias = old_bias[keep_neurons]
            if new_bias.shape == model.layers[layer_idx].intermediate.bias.shape:
                model.layers[layer_idx].intermediate.bias.data.copy_(new_bias)

        old_key = f'bert.encoder.layer.{layer_idx}.output.dense.weight'
        if old_key in state_dict:
            old_weight = state_dict[old_key]
            new_weight = old_weight[:, keep_neurons]
            if new_weight.shape == model.layers[layer_idx].output.weight.shape:
                model.layers[layer_idx].output.weight.data.copy_(new_weight)

        old_key = f'bert.encoder.layer.{layer_idx}.output.dense.bias'
        if old_key in state_dict:
            old_bias = state_dict[old_key]
            if old_bias.shape == model.layers[layer_idx].output.bias.shape:
                model.layers[layer_idx].output.bias.data.copy_(old_bias)

        for norm_param in ['weight', 'bias']:
            old_key = f'bert.encoder.layer.{layer_idx}.output.LayerNorm.{norm_param}'
            if old_key in state_dict:
                old_param = state_dict[old_key]
                if old_param.shape == model.layers[layer_idx].output_norm.__getattr__(norm_param).shape:
                    model.layers[layer_idx].output_norm.__getattr__(norm_param).data.copy_(old_param)

    pooler_keys = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
    pooler_targets = ['pooler.weight', 'pooler.bias']
    for old_key, new_key in zip(pooler_keys, pooler_targets):
        if old_key in state_dict and new_key in model.state_dict():
            model.state_dict()[new_key].copy_(state_dict[old_key])

    classifier_keys = ['classifier.weight', 'classifier.bias']
    for key in classifier_keys:
        if key in state_dict and key in model.state_dict():
            model.state_dict()[key].copy_(state_dict[key])


def getPrunedModel(base_dir: str):
    try:
        head_mask_path = f"{base_dir}/head_mask.pt"
        neuron_mask_path = f"{base_dir}/neuron_mask.pt"
        model_path = f"{'/'.join((base_dir.replace('outputs', 'pretrained').split('/'))[:3])}/pytorch_model.bin"

        head_mask = torch.load(head_mask_path).bool()
        neuron_mask = torch.load(neuron_mask_path).bool()

        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        model = PrunedBertForSequenceClassification(config, head_mask, neuron_mask)

        state_dict = torch.load(model_path, map_location='cpu')
        load_and_transfer_weights(model, state_dict, head_mask, neuron_mask)

        test_input = torch.randint(0, 1000, (2, 10))
        with torch.no_grad():
            _ = model(test_input)

        return model

    except FileNotFoundError as e:
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
