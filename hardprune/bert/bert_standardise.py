import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from hardPrune.qnli.qnli_prune_model import getPrunedModel
from transformers.models.bert.modeling_bert import BertEmbeddings

class StandardBertSelfAttention(nn.Module):
    def __init__(self, config, attention_dim=768):
        super().__init__()
        self.config = config
        self.attention_dim = attention_dim
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = attention_dim // self.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, attention_dim)
        self.key = nn.Linear(config.hidden_size, attention_dim)
        self.value = nn.Linear(config.hidden_size, attention_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_dim,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class StandardBertSelfOutput(nn.Module):
    def __init__(self, config, attention_dim=768):
        super().__init__()
        self.dense = nn.Linear(attention_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class StandardBertAttention(nn.Module):
    def __init__(self, config, attention_dim=768):
        super().__init__()
        self.self = StandardBertSelfAttention(config, attention_dim)
        self.output = StandardBertSelfOutput(config, attention_dim)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class StandardBertIntermediate(nn.Module):
    def __init__(self, config, intermediate_size=3072):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class StandardBertOutput(nn.Module):
    def __init__(self, config, intermediate_size=3072):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class StandardBertLayer(nn.Module):
    def __init__(self, config, attention_dim=768, intermediate_size=3072):
        super().__init__()
        self.attention = StandardBertAttention(config, attention_dim)
        self.intermediate = StandardBertIntermediate(config, intermediate_size)
        self.output = StandardBertOutput(config, intermediate_size)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class StandardBertEncoder(nn.Module):
    def __init__(self, config, layer_configs=None):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList()

        if layer_configs is None:
            layer_configs = [{'attention_dim': 768, 'intermediate_size': 3072} 
                           for _ in range(config.num_hidden_layers)]

        for i in range(config.num_hidden_layers):
            layer_config = layer_configs[i]
            layer = StandardBertLayer(
                config, 
                attention_dim=layer_config['attention_dim'],
                intermediate_size=layer_config['intermediate_size']
            )
            self.layer.append(layer)

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class StandardBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class StandardBertModel(nn.Module):
    def __init__(self, config, layer_configs=None):
        super().__init__()
        self.config = config
        bert_model = BertModel(config)
        self.embeddings = bert_model.embeddings
        self.encoder = StandardBertEncoder(config, layer_configs)
        self.pooler = StandardBertPooler(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs)

        return {
            'last_hidden_state': encoder_outputs,
            'pooler_output': pooled_output
        }

class StandardBertForSequenceClassification(nn.Module):
    def __init__(self, config, layer_configs=None):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.bert = StandardBertModel(config, layer_configs)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
            'last_hidden_state': outputs['last_hidden_state'],
            'pooler_output': pooled_output
        }

def extract_layer_configs_from_pruned_model(pruned_model):
    layer_configs = []
    for i, layer in enumerate(pruned_model.layers):
        attention_dim = layer.query.out_features
        intermediate_size = layer.intermediate.out_features
        config = {
            'attention_dim': attention_dim,
            'intermediate_size': intermediate_size
        }
        layer_configs.append(config)
    return layer_configs

def convert_pruned_to_standard_bert(pruned_model, config=None):
    if config is None:
        config = pruned_model.config
    layer_configs = extract_layer_configs_from_pruned_model(pruned_model)
    standard_model = StandardBertForSequenceClassification(config, layer_configs)
    transfer_weights_from_pruned(pruned_model, standard_model)
    return standard_model

def transfer_weights_from_pruned(pruned_model, standard_model):
    standard_model.bert.embeddings.load_state_dict(pruned_model.embeddings.state_dict())
    standard_model.classifier.weight.data.copy_(pruned_model.classifier.weight.data)
    standard_model.classifier.bias.data.copy_(pruned_model.classifier.bias.data)
    standard_model.bert.pooler.dense.weight.data.copy_(pruned_model.pooler.weight.data)
    standard_model.bert.pooler.dense.bias.data.copy_(pruned_model.pooler.bias.data)

    for i, (pruned_layer, standard_layer) in enumerate(zip(pruned_model.layers, standard_model.bert.encoder.layer)):
        standard_layer.attention.self.query.weight.data.copy_(pruned_layer.query.weight.data)
        standard_layer.attention.self.query.bias.data.copy_(pruned_layer.query.bias.data)
        standard_layer.attention.self.key.weight.data.copy_(pruned_layer.key.weight.data)
        standard_layer.attention.self.key.bias.data.copy_(pruned_layer.key.bias.data)
        standard_layer.attention.self.value.weight.data.copy_(pruned_layer.value.weight.data)
        standard_layer.attention.self.value.bias.data.copy_(pruned_layer.value.bias.data)
        standard_layer.attention.output.dense.weight.data.copy_(pruned_layer.att_out.weight.data)
        standard_layer.attention.output.dense.bias.data.copy_(pruned_layer.att_out.bias.data)
        standard_layer.attention.output.LayerNorm.weight.data.copy_(pruned_layer.att_norm.weight.data)
        standard_layer.attention.output.LayerNorm.bias.data.copy_(pruned_layer.att_norm.bias.data)
        standard_layer.intermediate.dense.weight.data.copy_(pruned_layer.intermediate.weight.data)
        standard_layer.intermediate.dense.bias.data.copy_(pruned_layer.intermediate.bias.data)
        standard_layer.output.dense.weight.data.copy_(pruned_layer.output.weight.data)
        standard_layer.output.dense.bias.data.copy_(pruned_layer.output.bias.data)
        standard_layer.output.LayerNorm.weight.data.copy_(pruned_layer.output_norm.weight.data)
        standard_layer.output.LayerNorm.bias.data.copy_(pruned_layer.output_norm.bias.data)

def verify_model_structure(model):
    expected_modules = ['bert', 'dropout', 'classifier']
    for module_name in expected_modules:
        if not hasattr(model, module_name):
            raise ValueError(f"缺少 {module_name}")

    if hasattr(model, 'bert'):
        bert_modules = ['embeddings', 'encoder', 'pooler']
        for module_name in bert_modules:
            if not hasattr(model.bert, module_name):
                raise ValueError(f"缺少 bert.{module_name}")

    if hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
        num_layers = len(model.bert.encoder.layer)
        if num_layers > 0:
            first_layer = model.bert.encoder.layer[0]
            layer_modules = ['attention', 'intermediate', 'output']
            for module_name in layer_modules:
                if not hasattr(first_layer, module_name):
                    raise ValueError(f"缺少 layer.{module_name}")

def getStandardModel(base_dir: str):
    pruned_model = getPrunedModel(base_dir)
    standard_model = convert_pruned_to_standard_bert(pruned_model)
    return standard_model
