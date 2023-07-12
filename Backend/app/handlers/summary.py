# summarize text using extractive summarization

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("punkt")


def preprocess_text(text: str):
    # tokenize text
    sentences = sent_tokenize(text)

    # get stopwords
    stop_words = stopwords.words("english")

    # preprocess text
    processed_sentences = []
    for sentence in sentences:
        # remove special characters and convert to lowercase
        sentence = "".join(c for c in sentence if c.isalnum() or c.isspace())
        sentence = sentence.lower()

        # tokenize sentence
        words = word_tokenize(sentence)

        # remove stopwords and stem words
        words = [word for word in words if word not in stop_words]

        # concatenate words to form sentence
        sentence = " ".join(words)

        # append to processed sentences
        processed_sentences.append(sentence)

    return processed_sentences


def generate_summary(text: str):
    # preprocess text
    processed_sentences = preprocess_text(text)

    # create tf-idf vectorizer
    vectorizer = TfidfVectorizer()

    # create tf-idf matrix
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # calculate similarity scores
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)

    # get index of most similar sentence excluding last sentence
    most_similar_sentence_index = similarity_scores.argsort()[0][-2]

    # get most similar sentence
    most_similar_sentence = processed_sentences[most_similar_sentence_index]

    return most_similar_sentence
