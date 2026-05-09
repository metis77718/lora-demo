from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")

lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, lora_config)

train_texts = [
    "To be, or not to be: that is the question.",
    "Whether 'tis nobler in the mind to suffer",
    "The slings and arrows of outrageous fortune,",
    "Or to take arms against a sea of troubles,",
    "And by opposing end them?"
]
encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")

# Создаём датасет с явными labels
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"]  # добавляем labels
        return item

dataset = SimpleDataset(encodings)

training_args = TrainingArguments(
    output_dir="./gpt2-lora-results",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    logging_steps=1,
    save_steps=500,
    fp16=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained("./gpt2-lora-adapter")
tokenizer.save_pretrained("./gpt2-lora-adapter")
print("LoRA adapter saved to ./gpt2-lora-adapter")
