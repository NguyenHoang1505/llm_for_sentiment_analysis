import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from langchain_openai import ChatOpenAI

# Load your training data
train_data = pd.read_csv("processed_data/train_data.csv")

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=""
)

# Function to generate a prompt for evaluation data
def generate_test_prompt(data_point, num_samples):
    return f"""Đây là đánh giá sản phẩm của khách hàng. Bạn hãy tạo ra {num_samples} đánh giá tương tự, giữ nguyên cảm xúc của khách hàng, giữ nguyên loại sản phẩm: '{data_point["text"]}'. Được biết cảm xúc của câu ban đầu là {data_point["sentiment"]}.""".strip()

# Function to extract generated samples from LLM response
def extract_samples(response_text):
    # This pattern looks for numbered items (1., 2., etc.) or items separated by newlines
    samples = re.split(r'\d+\.\s*|\n\s*-\s*|\n\n', response_text)
    # Filter out empty strings and strip whitespace
    samples = [sample.strip() for sample in samples if sample.strip()]
    
    # Remove quotes at the beginning and end of each sample
    cleaned_samples = []
    for sample in samples:
        # Remove leading and trailing single or double quotes
        sample = re.sub(r'^[\'"]|[\'"]$', '', sample)
        # If there are still pairs of quotes at both ends, remove them too
        sample = re.sub(r'^[\'"](.+)[\'"]$', r'\1', sample)
        cleaned_samples.append(sample)
    
    return cleaned_samples

# Create a new DataFrame for the augmented data
augmented_data = pd.DataFrame(columns=train_data.columns)

# Number of samples to generate for each training example
num_samples_to_generate = 3

# Loop through a subset of the training data (adjust as needed)
sample_size = 1000  # Adjust based on your needs and API usage limits
for i in tqdm(range(min(sample_size, len(train_data)))):
    data_point = train_data.iloc[i]
    prompt = generate_test_prompt(data_point, num_samples_to_generate)
    
    try:
        # Generate text using LLM
        messages = [("human", prompt)]
        ai_msg = llm.invoke(messages)
        
        # Extract the generated samples
        generated_samples = extract_samples(ai_msg.content)
        
        # Create new rows with the same sentiment as the original
        for sample in generated_samples:
            new_row = data_point.copy()
            new_row['text'] = sample
            # Convert to DataFrame and append to augmented data
            augmented_data = pd.concat([augmented_data, pd.DataFrame([new_row])], ignore_index=True)
            
    except Exception as e:
        print(f"Error processing item {i}: {e}")

# Combine original training data with augmented data
combined_data = pd.concat([train_data, augmented_data], ignore_index=True)

# Shuffle the combined dataset
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Save the augmented dataset
combined_data.to_csv("processed_data/train_data_augmented.csv", index=False)

print(f"Original dataset size: {len(train_data)}")
print(f"Augmented samples generated: {len(augmented_data)}")
print(f"Total combined dataset size: {len(combined_data)}")