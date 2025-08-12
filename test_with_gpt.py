import os
import argparse
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from evaluation import evaluate  # uses your existing evaluate
# NOTE: we avoid shadowing any existing `predict` you might have elsewhere.


PROMPT_TEMPLATE = (
    "Đây là đánh giá sản phẩm của khách hàng. Bạn hãy chỉ ra cảm xúc của khách hàng "
    "trong đánh giá sau: '{text}' là\n\n"
    " Tích cực\n Tiêu cực\n Trung lập\n Không thể xác định\n\n"
    "Đáp án: Lựa chọn đúng là"
)


def make_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(text=text)


def map_answer_to_label(answer: str) -> str:
    a = answer.lower()
    # keep order specific to avoid substring conflicts
    if "tích cực" in a:
        return "tích cực"
    if "tiêu cực" in a:
        return "tiêu cực"
    if "trung lập" in a:
        return "trung lập"
    if "không thể xác định" in a or "khong the xac dinh" in a:
        return "không thể xác định"
    return "none"


def parse_model_output(text: str) -> str:
    # take everything after the keyword if present, else the whole string
    after = text.split("Lựa chọn đúng là", 1)
    tail = after[-1] if len(after) > 1 else text
    # strip quotes/colons/whitespace
    tail = tail.strip().lstrip(":").strip()
    tail = re.sub(r'^[\'"]|[\'"]$', "", tail).strip()
    return map_answer_to_label(tail)


def build_test_df(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    y_true = df["sentiment"]
    X_test = pd.DataFrame({"text": df["text"].apply(make_prompt)})
    return X_test, y_true


def init_llm(model: str, temperature: float, api_key: str | None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=2,
        api_key=api_key,
    )


def predict_with_llm_batched(
    X_test: pd.DataFrame,
    llm: ChatOpenAI,
    batch_size: int = 16,
    show_outputs: bool = False,
) -> List[str]:
    messages = [[("human", prompt)] for prompt in X_test["text"].tolist()]
    preds: List[str] = []

    for i in tqdm(range(0, len(messages), batch_size), desc="Inferencing"):
        chunk = messages[i : i + batch_size]
        # ChatOpenAI supports .batch for parallel calls
        try:
            responses = llm.batch(chunk)
        except Exception:
            # fallback to sequential if batch not supported in your runtime
            responses = [llm.invoke(m) for m in chunk]

        for resp in responses:
            content = getattr(resp, "content", str(resp))
            if show_outputs:
                print(content)
            preds.append(parse_model_output(content))

    return preds


def save_numpy(path: str, arr: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.array(arr, dtype=object))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM sentiment predictions.")
    parser.add_argument("--test_csv", default="processed_data/test_data.csv")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_npy", default="test_with_gpt.npy")
    parser.add_argument("--show_outputs", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("Warning: No API key provided. Set --api_key or OPENAI_API_KEY.")

    X_test, y_true = build_test_df(args.test_csv)
    llm = init_llm(args.model, args.temperature, args.api_key)
    y_pred = predict_with_llm_batched(
        X_test=X_test, llm=llm, batch_size=args.batch_size, show_outputs=args.show_outputs
    )

    save_numpy(args.save_npy, y_pred)
    evaluate(y_true, y_pred)  # uses your existing signature
    print(f"Saved predictions to: {args.save_npy}")


if __name__ == "__main__":
    main()
