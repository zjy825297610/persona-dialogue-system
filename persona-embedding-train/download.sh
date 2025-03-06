# download

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download FacebookAI/xlm-roberta-base --local-dir oringinal/roberta_base
huggingface-cli download --resume-download BAAI/bge-m3 --local-dir oringinal/bge_base
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir oringinal/bert_base
huggingface-cli download --resume-download openai-community/gpt2 --local-dir oringinal/gpt2
huggingface-cli download --resume-download Qwen/Qwen2-0.5B --local-dir oringinal/qwen2