# mist.llm (wip)
A plug-and-play module for deploying MIST to LLM training. 


## Usage
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from datasets import load_dataset
from MIST import MISTTrainer

# Load and preprocess dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded_dataset = dataset.map(lambda x: tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True), batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model and set up training arguments
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", per_device_train_batch_size=8, num_train_epochs=2)

# Initialize and train with MISTTrainer
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
trained_model = mist_trainer.train()
```
