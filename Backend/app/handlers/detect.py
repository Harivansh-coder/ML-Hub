# route handler for /detect endpoint

# import dependencies
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from pathlib import Path

# Get the path to the directory of this script
script_dir = Path(__file__).resolve().parent


# load the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)


# load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


# Load the trained model
model_file = script_dir / "saved_model" / "bert_model.pt"

try:
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
except:
    raise Exception("Model not found")


# Define the prediction function
def predict(text):
    model.eval()
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]
    outputs = model(input_ids, attention_mask)
    _, prediction = torch.max(outputs.logits, dim=1)
    return prediction.item()


if __name__ == "__main__":
    print(predict("this is an ai content written by a human being"))
    print(predict("I hate you"))
