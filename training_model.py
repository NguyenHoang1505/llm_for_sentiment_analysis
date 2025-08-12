import time
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)


def generate_train_prompt(row: pd.Series) -> str:
    """Build the supervised fine-tuning prompt from a row with 'text' and 'sentiment'."""
    return (
        "Đây là đánh giá sản phẩm của khách hàng. Bạn hãy chỉ ra cảm xúc của khách hàng "
        f"trong đánh giá sau: '{row['text']}' là\n"
        "            \n\n Tích cực\n"
        "            \n Tiêu cực\n"
        "            \n Trung lập\n"
        "            \n Không thể xác định\n"
        f"            \n\nĐáp án: Lựa chọn đúng là {row['sentiment']}"
    ).strip()


def load_training_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    prompts = pd.DataFrame(df.apply(generate_train_prompt, axis=1), columns=["text"])
    return Dataset.from_pandas(prompts)


def load_model_and_tokenizer(model_name: str, access_token: str = ""):
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        token=access_token or None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=access_token or None,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def make_trainer(model, tokenizer, train_dataset: Dataset) -> SFTTrainer:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir="logs",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        max_seq_length=512,
    )
    return trainer


def main():
    csv_path = "processed_data/train_data.csv"
    model_name = "Viet-Mistral/Vistral-7B-Chat"
    access_token = ""  # set if needed for private hub access

    train_dataset = load_training_dataset(csv_path)
    model, tokenizer = load_model_and_tokenizer(model_name, access_token)
    trainer = make_trainer(model, tokenizer, train_dataset)

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    trainer.model.save_pretrained("models/trained-model")


if __name__ == "__main__":
    main()
