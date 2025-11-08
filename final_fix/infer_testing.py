import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "./final_kto_model_merged"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    adapter_path,
)
model.eval()

print("Base model loaded.")

def generate_clarification(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates clarifying questions to gain insights on the user's intent."},
        {"role": "user", "content": question},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_clarification("Who played beast in the beauty and the beast tv show?"))
print(generate_clarification("Who was USA president in 2015?"))
