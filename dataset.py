# Restricted dataset size by the amount of memory available on google colab
from datasets import load_dataset

train_dataset = load_dataset("allenai/peS2o", split="train[:70%]") 
valid_dataset = load_dataset("allenai/peS2o", split="valid[:70%]") 

train_data = train_dataset["train"]
valid_data = valid_dataset["valid"]

print("Train Subset Info:", train_data.info)
print("Valid Subset Info:", valid_data.info)
