# # route handler for /detect endpoint

# # import dependencies
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer

# from pathlib import Path

# # Get the path to the directory of this script
# script_dir = Path(__file__).resolve().parent


# # load the model
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=2,
#     output_attentions=False,
#     output_hidden_states=False,
# )


# # load the tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


# # Load the trained model
# model_file = script_dir / "saved_model" / "bert_model.pt"

# try:
#     model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
# except:
#     raise Exception("Model not found")


# # Define the prediction function
# def predict_ai_content(text):
#     model.eval()
#     encoded_dict = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=64,
#         padding="max_length",
#         return_attention_mask=True,
#         return_tensors="pt",
#     )

#     input_ids = encoded_dict["input_ids"]
#     attention_mask = encoded_dict["attention_mask"]
#     outputs = model(input_ids, attention_mask)
#     _, prediction = torch.max(outputs.logits, dim=1)
#     return prediction.item()


# if __name__ == "__main__":
#     print(predict_ai_content("this is an ai content written by a human being"))
#     print(predict_ai_content("I hate you"))


# import torch
# from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# # Get the path to the directory of this script
# script_dir = Path(__file__).resolve().parent

# # Sample data for AI content detection (replace this with your own labeled data)
# ai_content_data = [
#     ("This is an article about AI.", 1),  # 1 indicates AI content
#     ("Python programming is great.", 0),  # 0 indicates non-AI content
#     ("Machine learning is a subset of AI.", 1),
#     ("I love natural language processing.", 0),
#     ("AI is our friend and it has been friendly.", 1),
#     ("The sky is blue.", 0),
#     ("AI is used in many applications.", 1),
#     ("AI is a hot topic nowadays.", 1),
#     ("I hate you.", 0),
#     ("I love you.", 0),
#     ("AI is the new electricity.", 1),
#     ("I love Python.", 0),
#     ("I love Java.", 0),
#     ("AI can be used to solve many problems.", 1),
#     ("AI is the best.", 1),
#     # Add more samples of AI and non-AI content
# ]

# # Split data into sentences and labels
# sentences, labels = zip(*ai_content_data)

# # Split data into training and testing sets
# train_sentences, test_sentences, train_labels, test_labels = train_test_split(
#     sentences, labels, test_size=0.2, random_state=42
# )

# # Load the DistilBERT tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained(
#     "distilbert-base-uncased", do_lower_case=True
# )

# # Tokenize the data
# train_encodings = tokenizer(list(train_sentences), truncation=True, padding=True)
# test_encodings = tokenizer(list(test_sentences), truncation=True, padding=True)

# # Convert the encodings to PyTorch tensors
# train_input_ids = torch.tensor(train_encodings["input_ids"])
# train_attention_mask = torch.tensor(train_encodings["attention_mask"])
# train_labels = torch.tensor(train_labels)

# test_input_ids = torch.tensor(test_encodings["input_ids"])
# test_attention_mask = torch.tensor(test_encodings["attention_mask"])

# # Load the DistilBERT model
# model = DistilBertForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased", num_labels=2
# )

# # Define the training parameters
# train_batch_size = 8
# num_epochs = 3
# learning_rate = 2e-5

# # Create the data loader for training
# train_dataset = torch.utils.data.TensorDataset(
#     train_input_ids, train_attention_mask, train_labels
# )
# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=train_batch_size, shuffle=True
# )

# # Define the optimizer and loss function
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# loss_fn = torch.nn.CrossEntropyLoss()

# # Train the model
# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         optimizer.zero_grad()
#         input_ids, attention_mask, labels = batch
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = loss_fn(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()

# # Save the trained model
# output_model_file = script_dir / "saved_model" / "distilbert_model.pt"
# torch.save(model.state_dict(), output_model_file)


# # Define the prediction function
# def predict_ai_content(text):
#     model.eval()
#     encoded_dict = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=64,
#         padding="max_length",
#         return_attention_mask=True,
#         return_tensors="pt",
#     )

#     input_ids = encoded_dict["input_ids"]
#     attention_mask = encoded_dict["attention_mask"]
#     outputs = model(input_ids, attention_mask)
#     _, prediction = torch.max(outputs.logits, dim=1)
#     return prediction.item()


# # Test the AI content detection function
# text1 = "This is an article about AI."
# text2 = "I love natural language processing."
# print("Is text1 AI content?", predict_ai_content(text1))
# print("Is text2 AI content?", predict_ai_content(text2))
