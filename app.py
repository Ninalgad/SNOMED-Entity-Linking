import torch
import pandas as pd
from flask import Flask, jsonify, render_template
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer

from segmentation import segment


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESH = 0.5714285714285714

EMB_FILE = 'vectors.kv'

DEBUG = False

if not DEBUG:
    SEG_MODEL = AutoModelForTokenClassification.from_pretrained('xlreator/snomed-canine-s')
    SEG_TOKENIZER = AutoTokenizer.from_pretrained("google/canine-s")
    EMB_MODEL = SentenceTransformer('xlreator/biosyn-biobert-snomed')


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
        t = f'<div class="tooltip" style="background-color:{row.color};">{t}<span class="tooltiptext">{row.concept}</span></div>'

        highlighted += text[pointer:row.start] + t
        pointer = row.end
    highlighted += text[pointer:]

    # fix alignment bug
    highlighted += '<br><br><br><br>'

    return highlighted


app = Flask(__name__)
linker = Linker()


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/search/<query>', methods=['GET', 'POST'])
def entity_link(query):
    if DEBUG:
        result = {'highlight': "Of 1035 records entered into the CCC19 database during the study period, 928 patients met inclusion criteria for our analysis. <mark class='container__mark' style='background-color:rgba(19, 104, 53, .5);'>Median</mark> age was 66 years (IQR 57–76), 279 (30%) were aged 75 years or older, and 468 (50%) patients were male. The most prevalent <mark class='container__mark' style='background-color:rgba(105, 151, 37, .5);'>malignancies</mark> were <mark class='container__mark' style='background-color:rgba(151, 118, 0, .5);'>breast</mark> (191 [21%]) and <mark class='container__mark' style='background-color:rgba(23, 119, 153, .5);'>prostate</mark> (152 [16%]). 366 (39%) patients were on active <mark class='container__mark' style='background-color:rgba(32, 101, 135, .5);'>anticancer</mark> trea<mark class='container__mark' style='background-color:rgba(133, 56, 88, .5);'>tment</mark>, and 396 (43%) had active (measurable) <mark class='container__mark' style='background-color:rgba(105, 151, 37, .5);'>cancer</mark>. At analysis (May 7, 2020), 121 (13%) patients had <mark class='container__mark' style='background-color:rgba(35, 88, 135, .5);'>died</mark>. In <mark class='container__mark' style='background-color:rgba(145, 150, 128, .5);'>logistic regression</mark> analysis, independent factors associated with <mark class='container__mark' style='background-color:rgba(87, 144, 32, .5);'>increased</mark> 30-day mortality, after partial adjustment, were: <mark class='container__mark' style='background-color:rgba(87, 144, 32, .5);'>increased </mark>ag<mark class='container__mark' style='background-color:rgba(73, 100, 84, .5);'>e</mark> (per 10 years; partially adjusted odds ratio 1·84, 95% CI 1·53–2·21), <mark class='container__mark' style='background-color:rgba(48, 39, 33, .5);'>male sex</mark> (1·63, 1·07–2·48), <mark class='container__mark' style='background-color:rgba(54, 132, 65, .5);'>smoking</mark> status (former <mark class='container__mark' style='background-color:rgba(54, 132, 65, .5);'>smoker</mark> vs never <mark class='container__mark' style='background-color:rgba(118, 34, 2, .5);'>smoked</mark>: 1·60, 1·03–2·47), number of <mark class='container__mark' style='background-color:rgba(37, 148, 73, .5);'>comorbidities</mark> (two vs none: 4·50, 1·33–15·28), Eastern Cooperative Oncology Group performance status of 2 or higher (status of 2 vs 0 or 1: 3·89, 2·11–7·18), <mark class='container__mark' style='background-color:rgba(105, 151, 37, .5);'>active cancer</mark> (progressing vs <mark class='container__mark' style='background-color:rgba(39, 16, 64, .5);'>remission</mark>: 5·20, 2·77–9·77), and receipt of azithromycin plus hydroxychloroquine (vs treatment with neither: 2·93, 1·79–4·79; <mark class='container__mark' style='background-color:rgba(37, 148, 73, .5);'>confounding</mark> by indication cannot be excluded). Compared with residence in the US-Northeast, residence in Canada (0·24, 0·07–0·84) or the US-Midwest (0·50, 0·28–0·90) were associated with <mark class='container__mark' style='background-color:rgba(87, 38, 68, .5);'>decreased</mark> 30-day <mark class='container__mark' style='background-color:rgba(35, 88, 135, .5);'>all-cause mortality</mark>. Race and ethnicity, <mark class='container__mark' style='background-color:rgba(128, 34, 114, .5);'>obesity status</mark>, <mark class='container__mark' style='background-color:rgba(105, 151, 37, .5);'>cancer</mark> type, type of <mark class='container__mark' style='background-color:rgba(51, 71, 70, .5);'>anticancer therapy</mark>, and recent <mark class='container__mark' style='background-color:rgba(57, 52, 3, .5);'>surgery</mark> were not associated with <mark class='container__mark' style='background-color:rgba(99, 5, 51, .5);'>mort</mark>ali<mark class='container__mark' style='background-color:rgba(88, 6, 104, .5);'>ty</mark>.<br><br><br><br>", 'tab': [{'concept': 'quantitative measurement of substance ratio of collagen cross linked n-telopeptide in relation to creatinine in 24 hour urine specimen (procedure)', 'color': 'rgba(19, 104, 53, .5)'}, {'concept': 'malignant neoplastic disease (disorder)', 'color': 'rgba(105, 151, 37, .5)'}, {'concept': 'breast structure (body structure)', 'color': 'rgba(151, 118, 0, .5)'}, {'concept': 'prostatic structure (body structure)', 'color': 'rgba(23, 119, 153, .5)'}, {'concept': 'chemotherapy for malignant neoplastic disease using targeted agent (procedure)', 'color': 'rgba(32, 101, 135, .5)'}, {'concept': 'tho (body structure)', 'color': 'rgba(133, 56, 88, .5)'}, {'concept': 'died in hospital (finding)', 'color': 'rgba(35, 88, 135, .5)'}, {'concept': 'sequential multiple analysis - computer (procedure)', 'color': 'rgba(145, 150, 128, .5)'}, {'concept': 'increased hormone production (finding)', 'color': 'rgba(87, 144, 32, .5)'}, {'concept': 'hemoglobin e level (procedure)', 'color': 'rgba(73, 100, 84, .5)'}, {'concept': 'male (finding)', 'color': 'rgba(48, 39, 33, .5)'}, {'concept': 'smoker (finding)', 'color': 'rgba(54, 132, 65, .5)'}, {'concept': 'smokes tobacco daily (finding)', 'color': 'rgba(118, 34, 2, .5)'}, {'concept': 'co-morbid conditions (finding)', 'color': 'rgba(37, 148, 73, .5)'}, {'concept': 'disorder in remission (disorder)', 'color': 'rgba(39, 16, 64, .5)'}, {'concept': 'decreased general body function (finding)', 'color': 'rgba(87, 38, 68, .5)'}, {'concept': 'obesity (disorder)', 'color': 'rgba(128, 34, 114, .5)'}, {'concept': 'antineoplastic chemotherapy regimen (regime/therapy)', 'color': 'rgba(51, 71, 70, .5)'}, {'concept': 'surgical procedure (procedure)', 'color': 'rgba(57, 52, 3, .5)'}, {'concept': 'livor mortis (morphologic abnormality)', 'color': 'rgba(99, 5, 51, .5)'}, {'concept': 'infection caused by tyzzeria (disorder)', 'color': 'rgba(88, 6, 104, .5)'}]}
        return jsonify(error=None, result=result)

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
