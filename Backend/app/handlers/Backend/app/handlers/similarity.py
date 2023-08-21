# this file contains the similarity check handler

# import dependencies
from sentence_transformers import SentenceTransformer, util

# load the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Define the similarity check function
def check_text_similarity(text1, text2):
    # encode the texts
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    # compute cosine-similarities
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # output the similarity score
    return cosine_scores.item()


if __name__ == "__main__":
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast brown fox leaps over a sleeping canine."

    similarity_score = check_text_similarity(text1, text2)

    print(f"Similarity score: {similarity_score}")

    if similarity_score > 0.8:
        print("Similar")
    else:
        print("Not similar")
