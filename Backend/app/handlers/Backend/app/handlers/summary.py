# summarize text using extractive summarization

from transformers import BartTokenizer, BartForConditionalGeneration


def generate_abstractive_summary(text: str):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

    # Generate the summary
    summary_ids = model.generate(
        inputs.input_ids, num_beams=4, max_length=150, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
