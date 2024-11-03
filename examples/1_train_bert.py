from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from MIST import MISTTrainer

# Example usage with Hugging Face pipeline
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("glue", "mrpc")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")  # Rename to 'labels' for Trainer
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split into train and eval sets
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["validation"]

# Define training arguments
training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", num_train_epochs=3)

# Initialize and run MIST
mist_trainer = MISTTrainer(model, tokenized_dataset, num_submodels=4, split_name="train")
mist_trainer.train(training_args, eval_dataset=eval_dataset)
