import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# === CONFIG ===
#MODEL_NAME = "t5-base"
MODEL_NAME = "google/flan-t5-base"
TRAIN_FILE = "train.csv"
OUTPUT_DIR = "./flan_t5_gift_idea_finetuned"
BATCH_SIZE = 6
EPOCHS = 3
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 10

# === Load dataset ===
df = pd.read_csv(TRAIN_FILE)
dataset = Dataset.from_pandas(df)

# === Load tokenizer and model ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# === Preprocessing function ===
def preprocess_function(examples):
    inputs = ["simplify: " + title for title in examples["product_name"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

    labels = tokenizer(examples["simplified_idea"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    # Important: replace tokenizer.pad_token_id with -100 to ignore padding in loss
    model_inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in model_inputs["labels"]
    ]
    return model_inputs

# === Tokenize dataset ===
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_steps=100,
    report_to="none",
    push_to_hub=False,
)

# === Trainer setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# === Train! ===
trainer.train()

# === Save final model & tokenizer ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuned model saved to {OUTPUT_DIR}")
