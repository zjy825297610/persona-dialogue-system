import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, logging as hf_logging
from tools import read_from_json, save_as_json, save_results_excel, save_results_pic
from train import EncoderEmbeddingModel, DecoderEmbeddingModel
hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)

MODEL_LIST = [
    'bert_base',
    'bge_base',
    'gpt2',
    'qwen2',
    'roberta_base'
]

DATA_LIST = [
    'bad@3_test.jsonl',
    'bad@5_test.jsonl',
    'bad@7_test.jsonl',
    'bad@10_test.jsonl'
]


def caculate_acc(input_list):
    counter_mine = 0
    for line in input_list:
        if line == 1:
            counter_mine += 1
    return counter_mine / len(input_list)


def compute_metrics_single_query(q_emb, answers_emb_list, k):
    """
    给定一个query embedding 和一组(正确答案+负例)embeddings，计算多个指标。
    """
    candidates_emb = torch.stack(answers_emb_list, dim=0) 
    q_norm = q_emb / q_emb.norm(dim=-1, keepdim=True) 
    candidates_norm = candidates_emb / candidates_emb.norm(dim=-1, keepdim=True) 
    similarity = (q_norm * candidates_norm).sum(dim=-1)
    sorted_indices = torch.argsort(similarity, descending=True, dim=0) 
    rank_0_based = (sorted_indices == 0).nonzero(as_tuple=True)[0].item()
    top1_acc = 1.0 if rank_0_based == 0 else 0.0
    recall_at_k = 1.0 if rank_0_based < k else 0.0
    precision_at_k = (1.0 / k) if rank_0_based < k else 0.0
    mrr = 1.0 / (rank_0_based + 1)

    metrics_dict = {
        "top1_acc": top1_acc,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr
    }
    return metrics_dict


def compute_ndcg(q_emb, c_emb, k):
    """
    计算 nDCG@K
    假设第一个 candidate 是正确答案，其它全是负例。
    """
    c_emb_tensor = torch.cat(c_emb, dim=0) 
    q_norm = q_emb / q_emb.norm(dim=-1, keepdim=True)
    c_norm = c_emb_tensor / c_emb_tensor.norm(dim=-1, keepdim=True) 
    similarity_scores = torch.matmul(c_norm, q_norm.T).squeeze(-1) 
    relevance_scores = torch.zeros(len(c_emb), device='cuda')
    relevance_scores[0] = 1
    sorted_indices = torch.argsort(similarity_scores, descending=True)
    sorted_relevance = relevance_scores[sorted_indices]
    def dcg(scores, k_):
        scores = scores[:k_]
        return torch.sum(scores / torch.log2(torch.arange(2, scores.size(0) + 2, device='cuda').float()))

    actual_dcg = dcg(sorted_relevance, k)
    ideal_relevance = torch.sort(relevance_scores, descending=True).values
    ideal_dcg = dcg(ideal_relevance, k)
    ndcg = (actual_dcg / ideal_dcg).item() if ideal_dcg > 0 else 0.0
    return ndcg


def encode_texts(texts, model, tokenizer):
    """
    给定 texts (string 或 list)，使用 model + tokenizer 做推断。
    返回对应的 embedding 张量或 list。
    """
    if isinstance(texts, list):
        output_list = []
        for line in texts:
            cur_line = line[0]
            enc = tokenizer(cur_line[0], padding=True, truncation=True,
                            return_tensors="pt", max_length=128)
            enc = enc.to('cuda:0')
            with torch.no_grad():
                embeddings = model(enc["input_ids"], attention_mask=enc["attention_mask"])
            output_list.append(embeddings)
        return output_list
    else:
        enc = tokenizer(texts, padding=True, truncation=True,
                        return_tensors="pt", max_length=128)
        enc = enc.to('cuda:0')
        with torch.no_grad():
            embeddings = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        return embeddings


def compute_metrics(test_data, k, model, tokenizer):
    """
    给定数据集 test_data，计算各项指标并返回。
    """
    model.to('cuda:0')
    model.eval()

    recall_k_list = []
    precision_k_list = []
    mrr_list = []
    top1_acc_list = []
    nDCG_k_list = []

    for example in tqdm(test_data):
        question = example["question"]
        correct_answer = example["correct_answer"]
        negative_answers = example["negative_answers"]
        candidates = [correct_answer] + negative_answers
        q_emb = encode_texts(question, model, tokenizer) 
        c_emb = encode_texts(candidates, model, tokenizer) 
        metrics_dict = compute_metrics_single_query(q_emb, c_emb, k)
        top1_acc_list.append(metrics_dict['top1_acc'])
        recall_k_list.append(metrics_dict['recall_at_k'])
        precision_k_list.append(metrics_dict['precision_at_k'])
        mrr_list.append(metrics_dict['mrr'])
        nDCG_k_list.append(compute_ndcg(q_emb, c_emb, k))

    nDCG_k = np.mean(nDCG_k_list)
    top1Acc = caculate_acc(top1_acc_list)
    recall_k = np.mean(recall_k_list)
    precision_k = np.mean(precision_k_list)
    mrr = np.mean(mrr_list)

    return {
        f"Recall@{k}": round(recall_k * 100, 4),
        f"Precision@{k}": round(precision_k * 100, 4),
        "MRR": round(mrr * 100, 4),
        f"nDCG@{k}": round(nDCG_k * 100, 4),
        "Top1Acc": round(top1Acc * 100, 4)
    }


def create_test_data(K_NEG, json_path):
    """
    从原始文件生成新的带负例的测试集并保存。
    """
    test_data = read_from_json(json_path, False)
    all_answers = [item["answers"] for item in test_data]
    augmented_test_data = []

    import random
    for example in tqdm(test_data, desc='processing test data'):
        question = example["question"]
        correct_answer = example["answers"]
        negative_answers = random.sample(
            [ans for ans in all_answers if ans != correct_answer],
            K_NEG
        )
        augmented_test_data.append({
            "question": question,
            "correct_answer": correct_answer,
            "negative_answers": negative_answers
        })

    save_as_json(augmented_test_data, f'data/bad@{K_NEG}_test.jsonl')
    print('done')


def load_model_and_tokenizer(cur_model: str, baseline: bool):
    """
    根据是baseline还是finetuned，加载对应的模型 & tokenizer。
    返回 (model, tokenizer)
    """
    if baseline:
        model_path = f"oringinal/{cur_model}"
        if 'gpt2' in cur_model or 'qwen2' in cur_model:
            model = DecoderEmbeddingModel(model_path, is_config=False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if 'gpt2' in cur_model:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            model = EncoderEmbeddingModel(model_path, is_config=False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    else:
        model_path = f"finetune/{cur_model}/checkpoint-2745"
        if 'gpt2' in cur_model or 'qwen2' in cur_model:
            model = DecoderEmbeddingModel(model_path, is_config=False)
            tokenizer = AutoTokenizer.from_pretrained(f"oringinal/{cur_model}", use_fast=True)
            if 'gpt2' in cur_model:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            model = EncoderEmbeddingModel(model_path, is_config=False)
            tokenizer = AutoTokenizer.from_pretrained(f"oringinal/{cur_model}", use_fast=True)
    return model, tokenizer


def evaluate_model(cur_model: str, cur_data: str, baseline: bool) -> dict:
    """
    并行执行的核心函数：
    1) 加载模型(可能是baseline或finetuned)
    2) 读取数据
    3) 计算metrics
    4) 返回 {"模型|数据": metrics}
    """

    model, tokenizer = load_model_and_tokenizer(cur_model, baseline)
    test_data = read_from_json(f"data/{cur_data}", False)
    k_value = int(cur_data.split('_')[0].split('@')[-1])

    metrics = compute_metrics(test_data, k_value, model, tokenizer)

    if baseline:
        key_name = f"{cur_model}_baseline|{cur_data}"
    else:
        key_name = f"{cur_model}_finetuned|{cur_data}"

    return {key_name: metrics}


def main():
    cur_step = 1
    for cur_data in DATA_LIST:
        output_list = []
        logging.info(f"[{cur_step}/{len(DATA_LIST)}] eval in 「{cur_data}」")
        with ThreadPoolExecutor(max_workers=2*len(MODEL_LIST)) as executor:
            futures = []
            # 提交 baseline
            futures.append(
                executor.submit(evaluate_model, 'bert_base', cur_data, True)
            )
            futures.append(
                executor.submit(evaluate_model, 'bge_base', cur_data, True)
            )
            futures.append(
                executor.submit(evaluate_model, 'gpt2', cur_data, True)
            )
            futures.append(
                executor.submit(evaluate_model, 'qwen2', cur_data, True)
            )
            futures.append(
                executor.submit(evaluate_model, 'roberta_base', cur_data, True)
            )
            # 提交 finetuned
            futures.append(
                executor.submit(evaluate_model, 'bert_base', cur_data, False)
            )
            futures.append(
                executor.submit(evaluate_model, 'bge_base', cur_data, False)
            )
            futures.append(
                executor.submit(evaluate_model, 'gpt2', cur_data, False)
            )
            futures.append(
                executor.submit(evaluate_model, 'qwen2', cur_data, False)
            )
            futures.append(
                executor.submit(evaluate_model, 'roberta_base', cur_data, False)
            )

            for future in as_completed(futures):
                res_dict = future.result() 
                output_list.append(res_dict)
        cur_step += 1
        save_results_excel(output_list, f'results/{cur_data}_result.xlsx')
        save_results_pic(output_list, f'results/{cur_data}_result.png')


if __name__ == "__main__":
    main()