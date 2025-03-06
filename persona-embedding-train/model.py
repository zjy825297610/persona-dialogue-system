import torch
from torch import nn
from transformers import AutoModel, AutoConfig, GPT2Model, GPT2Config, AutoConfig, AutoModelForCausalLM

# 定义一个简单的model wrapper，返回sentence embeddings
class EncoderEmbeddingModel(nn.Module):
    def __init__(self, model_name:str, is_config:bool):
        super().__init__()
        if is_config:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        else:
            self.model = AutoModel.from_pretrained(model_name)
        # 可以根据需要冻结层或添加投影层
        # self.projection = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # last_hidden_state: [batch, seq_len, hidden_size]
        # pooler_output: [batch, hidden_size] (如果模型有pooler层，如BERT有)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            sentence_embeddings = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sentence_embeddings = torch.sum(last_hidden * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # 如果需要投影层：
        # sentence_embeddings = self.projection(sentence_embeddings)
        return sentence_embeddings
    
# 定义一个decoder的sentence embedding
class DecoderEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, is_config: bool=False):
        super().__init__()
        if "gpt2" in model_name:
            if is_config:
                config = AutoConfig.from_pretrained(model_name)
                self.model = GPT2Model(config)
            else:
                self.model = GPT2Model.from_pretrained(model_name)
        elif "qwen2" in model_name:
            if is_config:
                config = AutoConfig.from_pretrained(model_name)
                self.model = AutoModel(config)
            else:
                self.model = AutoModel.from_pretrained(model_name)

        # 可选：添加一层映射
        # self.projection = nn.Linear(self.model.config.n_embd, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        """
        输入:
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len], 用于区分实际token和padding(若有)
        
        输出:
        - sentence_embeddings: [batch, hidden_dim]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # GPT2Model输出: last_hidden_state: [batch, seq_len, hidden_size]
        # 我们需要自己设计如何pooling
        
        last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 方式1: 取最后一个 token 的向量作为句向量（类似 GPT2 做语言建模）
        # 注意，这里假设输入的句子已经满足 “最后一个token就是句子结束”，
        # 或者有 <|endoftext|> 之类的special token。
        # sentence_embeddings = last_hidden_state[:, -1, :]

        # 方式2: 对所有token做“平均池化”
        if attention_mask is not None:
            # 扩展mask: [batch, seq_len, 1] -> 和 hidden_state 逐元素相乘
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            # 若没有attention_mask, 简单对 seq_len 取平均
            sentence_embeddings = last_hidden_state.mean(dim=1)

        # 可选投影层
        # sentence_embeddings = self.projection(sentence_embeddings)

        return sentence_embeddings
    
# 定义一个decoder的sentence embedding
class AutoEncoderEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, is_config: bool=False):
        super().__init__()
        if is_config:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel(config)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        # 可选：添加一层映射
        # self.projection = nn.Linear(self.model.config.n_embd, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        """
        输入:
        - input_ids: [batch, seq_len]
        - attention_mask: [batch, seq_len], 用于区分实际token和padding(若有)
        
        输出:
        - sentence_embeddings: [batch, hidden_dim]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 方式2: 对所有token做“平均池化”
        if attention_mask is not None:
            # 扩展mask: [batch, seq_len, 1] -> 和 hidden_state 逐元素相乘
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
        else:
            # 若没有attention_mask, 简单对 seq_len 取平均
            sentence_embeddings = last_hidden_state.mean(dim=1)
        # 可选投影层
        # sentence_embeddings = self.projection(sentence_embeddings)
        return sentence_embeddings
