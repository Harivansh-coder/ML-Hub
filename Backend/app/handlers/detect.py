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





import tensorflow as tf
import time

# Define the size of the computation
matrix_size = 1000

# Create a TensorFlow graph
graph = tf.Graph()
with graph.as_default():
    # Create placeholder for input matrix
    input_matrix = tf.placeholder(tf.float32, shape=(matrix_size, matrix_size))

    # Perform a simple matrix multiplication
    output_matrix = tf.matmul(input_matrix, input_matrix)

# Create session and run the computation
with tf.Session(graph=graph) as sess:
    # Generate random input matrix
    input_data = tf.random.normal((matrix_size, matrix_size))

    # Benchmark GPU performance
    with tf.device('/gpu:0'):
        start_time = time.time()
        result_gpu = sess.run(output_matrix, feed_dict={input_matrix: input_data.eval()})
        gpu_time = time.time() - start_time
        print("GPU Time: %.2f seconds" % gpu_time)

    # Benchmark CPU performance
    with tf.device('/cpu:0'):
        start_time = time.time()
        result_cpu = sess.run(output_matrix, feed_dict={input_matrix: input_data.eval()})
        cpu_time = time.time() - start_time
        print("CPU Time: %.2f seconds" % cpu_time)

