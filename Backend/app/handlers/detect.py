# route handler for /detect endpoint

# import dependencies
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

# load the data fromt the csv file
import pandas as pd

df = pd.read_csv("./bert_dataset.csv")

# load the model
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=2,
#     output_attentions=False,
#     output_hidden_states=False,
# )

saved_model_path = "./model/data.pkl"
model = torch.load(saved_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# tokenize the data
input_ids = []
attention_masks = []

for sent in df["text"]:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])

# convert the lists into tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df["label"])

# create the dataset
dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 2
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define the optimizer and the learning rate
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 10

# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# define the training loop
from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")


# Classify new text
new_text = "AI is revolutionizing various industries"
inputs = tokenizer.encode_plus(
    new_text,
    add_special_tokens=True,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
outputs = model(input_ids, attention_mask)
predicted_label = torch.argmax(outputs.logits, dim=1).item()
print(
    f"Predicted label for new text: {predicted_label} (1 represents AI, 0 represents non-AI)"
)
