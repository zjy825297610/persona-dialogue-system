from model import EncoderEmbeddingModel, DecoderEmbeddingModel
from tools import read_from_json
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch.nn.functional as F
import random
import torch

# 用于统计模型参数大小的函数
def format_params(param_count):
    if param_count >= 1e8:
        value = param_count / 1e9
        unit = 'B'
    else:
        value = param_count / 1e6
        unit = 'M'
    return f"{param_count}({value:.1f}{unit})"

class SentenceEmbeddingTrainer(Trainer):
    # 重写loss函数
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        改进版：使用 In-batch Negatives + CrossEntropy，
        将同一 batch 内其他 (q, a) 对当作负例。
        """

        # 1) 取出 input_ids & attention_mask
        q_input_ids = inputs["q_input_ids"]         # [B, seq_len_q]
        q_attention_mask = inputs["q_attention_mask"]
        a_input_ids = inputs["a_input_ids"]         # [B, seq_len_a]
        a_attention_mask = inputs["a_attention_mask"]

        # 2) 用你的模型分别得到 query 和 answer 的表征
        #    假设 model(...) 返回 [B, dim]
        q_emb = model(q_input_ids, q_attention_mask)     # [B, dim]
        a_emb = model(a_input_ids, a_attention_mask)     # [B, dim]

        # 3) 归一化 embedding (方便用余弦相似度)
        q_emb_norm = F.normalize(q_emb, p=2, dim=-1)  # [B, dim]
        a_emb_norm = F.normalize(a_emb, p=2, dim=-1)  # [B, dim]

        # 4) 计算相似度矩阵 S: [B, B] = q_emb_norm @ a_emb_norm^T
        sim_matrix = torch.matmul(q_emb_norm, a_emb_norm.T)  # [B, B]

        # 5) 可加一个温度系数 tau 做缩放 (可调超参, 默认 0.07 / 0.05 等)
        temperature = 0.05
        sim_matrix = sim_matrix / temperature

        # 6) 构造 CrossEntropy 的标签：行的正确答案都是第 i 列
        B = sim_matrix.size(0)
        labels = torch.arange(B, device=sim_matrix.device)  # [0..B-1]

        # 7) 分别对行和列做 CE (对称式)
        #    (1) 行 -> 每行 i 的正确列 = i
        loss_q = F.cross_entropy(sim_matrix, labels)

        #    (2) 列 -> 每列 j 的正确行 = j
        #             sim_matrix.T 的 shape 也是 [B, B]
        loss_a = F.cross_entropy(sim_matrix.T, labels)

        # 8) 综合两个方向的损失
        loss = 0.5 * (loss_q + loss_a)

        # 如果你只想用单向监督 (query->answer)，就可以直接用 loss = loss_q

        # 9) 可以返回 (loss, (q_emb, a_emb)) 以便你在训练外部可访问
        return (loss, (q_emb, a_emb)) if return_outputs else loss

def do_train(model_name):
    # 需要进行训练，所以进行数据的打乱
    train_data = read_from_json('data/focus_tr_dataset.jsonl', True)
    random.shuffle(train_data)
    # 将数据转化为Hugging Face Dataset格式
    train_dataset = Dataset.from_list(train_data)

    model_name_or_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def preprocess(batch):
        q = batch["question"]
        a = batch["answers"][0]
        q_enc = tokenizer(q, truncation=True, padding="max_length", max_length=128)
        a_enc = tokenizer(a, truncation=True, padding="max_length", max_length=128)
        return {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "a_input_ids": a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"]
        }

    # decoder架构
    if "qwen2" in model_name_or_path:
        model = DecoderEmbeddingModel(model_name=model_name_or_path, is_config=False)
    if "gpt2" in model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        model = DecoderEmbeddingModel(model_name=model_name_or_path, is_config=False)
    # encoder架构
    else:
        model = EncoderEmbeddingModel(model_name=model_name_or_path, is_config=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{format_params(total_params)}")

    train_dataset = train_dataset.map(preprocess, batched=False)
    train_dataset = train_dataset.remove_columns(["question", "answers"])
    train_dataset.set_format(type="torch")

    training_args = TrainingArguments(
        # output_dir='finetune/' + model_name_or_path.split('/')[-1],
        output_dir='test',
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=10,
        remove_unused_columns=False,  # 很重要，避免Trainer删除不认识的输入列
        gradient_accumulation_steps=8 # 梯度累积
    )

    trainer = SentenceEmbeddingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    # 开始训练
    trainer.train()

    # 训练结束后，保存模型
    trainer.save_model("our_trained_models")


if __name__ == "__main__":
    # train_list = ['bert_base', 'bge_base', 'roberta_base', 'gpt2']
    # for model in train_list:
    #     do_train('oringinal/' + model)

    do_train('oringinal/qwen2')


# 6. 使用模型产生句向量
# 用trainer.model即可获得句向量，如：
# test_text = "What is this amazing thing?"
# enc = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
# with torch.no_grad():
#     embedding = model(**enc)
# embedding即为句子的向量表示z zaq

# openai-community/gpt2
# huggingface-cli download --resume-download openai-community/gpt2 --local-dir gpt2
# bigscience/bloom-560m
# huggingface-cli download --resume-download bigscience/bloom-560m --local-dir bloom
# Qwen/Qwen2-0.5B
# huggingface-cli download --resume-download Qwen/Qwen2-0.5B --local-dir qwen2