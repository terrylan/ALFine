import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Config
MODEL_NAME = "albert-base-v2"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
OUTPUT_DIR = "./fine_tuned_albert"

def main():
    # Load model and tokenizer
    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Load dataset
    dataset = load_dataset("imdb")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"]).set_format("torch")
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Fine-tuning complete. Model saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
