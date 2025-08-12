from tqdm import tqdm
from transformers import pipeline

def predict(X_test, model, tokenizer):
    # List to store predicted labels
    y_pred = []
    model_output = []
    # Loop through each data point in the test set
    for i in tqdm(range(len(X_test))):
        # Get the prompt from the test set
        prompt = X_test.iloc[i]["text"]

        # Create a text generation pipeline using the specified model and tokenizer
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=10,
                        temperature=0.0,
                       )

        # Generate text based on the prompt
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        # print(result)
        model_output.append(result[0]['generated_text'])
        # Extract the generated text and identify the sentiment label
        answer = result[0]['generated_text'].split("Lựa chọn đúng là")[-1].lower()
        if "tích cực" in answer:
            y_pred.append("tích cực")
        elif "tiêu cực" in answer:
            y_pred.append("tiêu cực")
        elif "trung lập" in answer:
            y_pred.append("trung lập")
        else:
            y_pred.append("none")

    return y_pred, model_output

