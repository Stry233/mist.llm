from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments

from core.mist_trainer import MISTTrainer


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

# Load optim for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Initialize MISTTrainer
mist_trainer = MISTTrainer(
    model=model,
    args=training_args,
    dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    num_local_models=3,
    T1=5,
    T2=2,
    cross_diff_weight=0.1,
    repartition=True
)

# Start training
mist_trainer.train()

