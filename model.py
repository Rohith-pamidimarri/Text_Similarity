import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityModel:
    def __init__(self, paraphrase_model_path="paraphrase-distilroberta-base-v1", accuracy_model_path="distilbert-base-nli-mean-tokens", weight_distilbert=0.6, weight_paraphrase=0.4, threshold=0.35):
        self.model_paraphrase = SentenceTransformer(paraphrase_model_path)
        self.model_accuracy = SentenceTransformer(accuracy_model_path)
        self.weight_distilbert = weight_distilbert
        self.weight_paraphrase = weight_paraphrase
        self.threshold = threshold

    def compute_similarity(self, sentence1, sentence2, model):
        embeddings = model.encode([sentence1, sentence2])
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(similarity_score, 4)

    def compute_score_based_on_threshold(self, sentence1, sentence2):
        similarity1 = self.compute_similarity(sentence1, sentence2, self.model_paraphrase)
        similarity2 = self.compute_similarity(sentence1, sentence2, self.model_accuracy)
        absolute_difference = abs(similarity1 - similarity2)

        if absolute_difference > self.threshold:
            return similarity1
        else:
            weighted_average_similarity = (similarity1 * self.weight_paraphrase + similarity2 * self.weight_distilbert) / (self.weight_paraphrase + self.weight_distilbert)
            return round(weighted_average_similarity, 4)

    def predict_similarity(self, sentence1, sentence2):
        ret_val=self.compute_score_based_on_threshold(sentence1, sentence2)
        if ret_val<0:
            return 0
        return ret_val


