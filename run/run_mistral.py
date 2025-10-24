from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto

# Define the local path
local_model_path = "model/Mistral-7B-Instruct-v0.2"

# Load the model and tokenizer from the local path
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    device_map="auto" # Recommended for large models
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
# No need for model.to(device) if you use device_map="auto"

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])