from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from model import SimilarityModel
import pickle

app = Flask(__name__)
# Create an instance of the SimilarityModel class
model = SimilarityModel(weight_distilbert=0.7,
                                   weight_paraphrase=0.3,
                                   threshold=0.38)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text1 = request.form['text1']
    text2 = request.form['text2']
    
    # Compute cosine similarity
    similarity_score = model.predict_similarity(text1,text2)
    
    return jsonify({"similarity_score": similarity_score})

if __name__ == '__main__':
    app.run(debug=True)
