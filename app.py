import pickle
from flask import Flask, jsonify, render_template
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification


SEG_MODEL = AutoModelForTokenClassification.from_pretrained('xlreator/snomed-canine-s')


GIF_FILE = '../server/data/href.pkl'
IMG_EMB_FILE = '../server/data/img-vectors.kv'
TXT_EMB_FILE = '../server/data/txt-vectors.kv'
SEG_MODEL = SentenceTransformer('clip-ViT-B-16')


class Linker:
    def __init__(self, topk=10):
        self.topk = topk
        self._image_vectors = None
        self._text_vectors = None
        self._links = None

    def _load_embeddings(self):
        self._image_vectors = KeyedVectors.load(IMG_EMB_FILE)
        self._text_vectors = KeyedVectors.load(TXT_EMB_FILE)

    def _load_gif_links(self):
        with open(GIF_FILE, 'rb') as f:
            self._links = pickle.load(f)

    @property
    def links(self):
        if self._links is None:
            self._load_gif_links()
        return self._links

    @property
    def embeddings(self):
        if self._image_vectors is None:
            self._load_embeddings()
        return {'img': self._image_vectors, 'text': self._text_vectors}

    def link(self, query: str) -> list[dict]:
        query_emb = EMB_MODEL.encode([query])
        similar_imgs = self.embeddings['img'].most_similar(query_emb, topn=self.topk)
        similar_txts = self.embeddings['text'].most_similar(query_emb, topn=self.topk)

        result = {'img': [{
                'emote': emote,
                'href': self.links[emote]['href'],
                'author': self.links[emote]['author'],
                'score': score
            } for (emote, score) in similar_imgs],
                'txt': [{
                    'emote': emote,
                    'href': self.links[emote]['href'],
                    'author': self.links[emote]['author'],
                    'score': score
                } for (emote, score) in similar_txts]}

        return result


app = Flask(__name__)
linker = Linker()


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/search/<query>', methods=['GET', 'POST'])
def entity_link(query):
    error = None
    result = []

    try:
        result = linker.link(query)
    except Exception as err:
        error = str(err)
    return jsonify(error=error, result=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
