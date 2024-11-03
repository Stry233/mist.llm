import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from transformers import TrainingArguments

from MIST import MISTTrainer

# Load dataset and tokenizer
dataset = load_dataset("glue", "mrpc")  # Replace with your dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocess function to align with BERT's expected input
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

# Tokenize the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Ensure "label" column is renamed to "labels" in encoded dataset
encoded_dataset = encoded_dataset.rename_column("label", "labels")

# Set format for PyTorch
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Training arguments (adjust as needed)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Initialize MISTTrainer with the model, dataset, and relevant parameters
mist_trainer = MISTTrainer(
    model=model,
    dataset=encoded_dataset["train"],
    num_local_models=2,
    T1=3,
    T2=3,
    cross_diff_weight=0.5,
    epochs=2,
    args=training_args
)

# Train the model
trained_model = mist_trainer.train()
