import torch
import pandas as pd
from flask import Flask, jsonify, render_template
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer

from segmentation import segment


SEG_MODEL = AutoModelForTokenClassification.from_pretrained('xlreator/snomed-canine-s')
SEG_TOKENIZER = AutoTokenizer.from_pretrained("google/canine-s")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESH = 0.5714285714285714

EMB_MODEL = SentenceTransformer('xlreator/biosyn-biobert-snomed')
EMB_FILE = 'vectors.kv'


def clean_entity(t):
    t = t.lower()
    t = t.replace(' \n', " ")
    t = t.replace('\n', " ")
    return t


def hash_to_rgb(hash_):
    # Ensure hash string is at least 6 characters long
    hash_str = str(abs(hash_))
    # Split hash into three parts and convert each to decimal
    r = int(hash_str[:2], 16)
    g = int(hash_str[2:4], 16)
    b = int(hash_str[4:6], 16)
    return f"rgba({r}, {g}, {b}, .5)"


class Linker:
    def __init__(self, context_window_width=18):
        self._vectors = None
        self.context_window_width = context_window_width

    def add_context(self, row):
        window_start = max(0, row.start - self.context_window_width)
        window_end = min(row.end + self.context_window_width, len(row.text))
        return clean_entity(row.text[window_start:window_end])

    def _load_embeddings(self):
        self._vectors = KeyedVectors.load(EMB_FILE)

    @property
    def embeddings(self):
        if self._vectors is None:
            self._load_embeddings()
        return self._vectors

    def link(self, df: pd.DataFrame) -> list[dict]:
        mention_emb = EMB_MODEL.encode(df.mention.str.lower().values)

        concepts = [self.embeddings.most_similar(m, topn=1)[0][0]
                    for m in mention_emb]
        return concepts


def insert_marker_html(spans, text):
    highlighted = ""
    pointer = 0

    for row in spans.itertuples():
        t = text[row.start: row.end]
        t = f"<mark class='container__mark' style='background-color:{row.color};'>{t}</mark>"

        highlighted += text[pointer:row.start] + t
        pointer = row.end
    highlighted += text[pointer:]

    return highlighted


app = Flask(__name__)
linker = Linker()


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/search/<query>', methods=['GET', 'POST'])
def entity_link(query):
    query_df = pd.DataFrame({'note_id': [0], 'text': [query]})
    error = None
    result = {}

    try:
        seg = segment(query_df, SEG_MODEL, SEG_TOKENIZER, DEVICE, THRESH)
        seg = seg.sort_values('start')
        linked_concepts = linker.link(seg)
        seg['concept'] = linked_concepts
        seg['color'] = [hash_to_rgb(hash(x)) for x in linked_concepts]

        result["highlight"] = insert_marker_html(seg, query)
        result["tab"] = seg[['concept', 'color']].drop_duplicates().to_dict(orient='records')
        print(result)
    except Exception as err:
        error = str(err)
    return jsonify(error=error, result=result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
