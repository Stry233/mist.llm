# mist.llm (wip)
A plug-and-play module for deploying MIST to LLM training. 


## Usage

```python
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from mist.core import MISTTrainer

# Load and preprocess dataset
dataset = load_dataset("glue", "mrpc").map(
   lambda x: tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True), batched=True)
dataset = dataset.rename_column("label", "labels").set_format("torch",
                                                              columns=["input_ids", "attention_mask", "labels"])

# Initialize model, arguments, and MISTTrainer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", num_train_epochs=3)

# Train!
mist_trainer = MISTTrainer(
   model=model,
   args=training_args,
   train_dataset=encoded_dataset["train"],
   eval_dataset=encoded_dataset["validation"],
   num_local_models=3,
   T1=5,
   T2=2,
   cross_diff_weight=0.1,
   repartition=True
)
mist_trainer.train()

```

## Install
To install `mist.llm`, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mist.llm.git
   cd mist.llm
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Paper
```
@inproceedings{li2024mist,
  title={$\{$MIST$\}$: Defending Against Membership Inference Attacks Through $\{$Membership-Invariant$\}$ Subspace Training},
  author={Li, Jiacheng and Li, Ninghui and Ribeiro, Bruno},
  booktitle={33rd USENIX Security Symposium (USENIX Security 24)},
  pages={2387--2404},
  year={2024}
}
```
