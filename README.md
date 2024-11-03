# mist.llm (wip)
A plug-and-play module for deploying MIST to LLM training. 


## Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Example usage with Hugging Face pipeline
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
dataset = load_dataset("glue", "mrpc")
training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", num_train_epochs=3)

# Initialize and run MIST
mist_trainer = MISTTrainer(model, dataset, num_submodels=4)
mist_trainer.train(training_args)
```
