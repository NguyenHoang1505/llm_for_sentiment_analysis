import os
import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from predict import predict
from evaluation import evaluate


def generate_test_prompt(row: pd.Series) -> str:
    return (
        "Đây là đánh giá sản phẩm của khách hàng. Bạn hãy chỉ ra cảm xúc của khách hàng "
        f"trong đánh giá sau: '{row['text']}' là\n\n Tích cực\n Tiêu cực\n Trung lập\n Không thể xác định\n\n"
        "Đáp án: Lựa chọn đúng là"
    ).strip()


def build_test_dataframe(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    prompts = pd.DataFrame(df.apply(generate_test_prompt, axis=1), columns=["text"])
    y_true = df["sentiment"]
    return prompts, y_true


def load_model_and_tokenizer(model_path: str, base_tokenizer: str, access_token: str = ""):
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        token=access_token or None,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer,
        trust_remote_code=True,
        token=access_token or None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def truncate_to_min(X_test: pd.DataFrame, y_true, y_pred, model_output):
    n = min(len(X_test), len(y_true), len(y_pred), len(model_output))
    return (
        X_test.iloc[:n].reset_index(drop=True),
        y_true.iloc[:n].reset_index(drop=True) if hasattr(y_true, "iloc") else y_true[:n],
        y_pred[:n],
        model_output[:n],
    )


def save_outputs(
    X_test: pd.DataFrame,
    y_true,
    y_pred,
    model_output,
    out_prefix: str,
):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    correct = [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]
    result = X_test.copy()
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    result["model_output"] = model_output
    result["correct"] = correct

    result.to_csv(f"{out_prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    result[result["correct"] == 1].to_csv(
        f"{out_prefix}_correct_predictions.csv", index=False, encoding="utf-8-sig"
    )
    result[result["correct"] == 0].to_csv(
        f"{out_prefix}_incorrect_predictions.csv", index=False, encoding="utf-8-sig"
    )
    return result


def main():
    test_csv = "processed_data/test_data.csv"
    model_path = "models/trained-model"
    base_tokenizer = "Viet-Mistral/Vistral-7B-Chat"
    access_token = ""  # set if needed
    exp_name = "exp05"
    out_prefix = f"output/{exp_name}"

    X_test, y_true = build_test_dataframe(test_csv)
    model, tokenizer = load_model_and_tokenizer(model_path, base_tokenizer, access_token)

    start = time.time()
    y_pred, model_output = predict(X_test, model, tokenizer)
    print(f"Inference time: {time.time() - start:.2f}s for {len(X_test)} rows")

    X_test, y_true, y_pred, model_output = truncate_to_min(X_test, y_true, y_pred, model_output)
    _ = save_outputs(X_test, y_true, y_pred, model_output, out_prefix)

    eval_path = f"{out_prefix}_evaluation.txt"
    evaluate(y_true, y_pred, output_file=eval_path)
    print(f"Saved predictions and evaluation to prefix: {out_prefix}")


if __name__ == "__main__":
    main()
