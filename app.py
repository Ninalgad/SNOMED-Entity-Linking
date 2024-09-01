import torch
import pandas as pd
import configparser
import gradio as gr
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForTokenClassification, AutoTokenizer

from segmentation import segment
from utils import clean_entity


class Linker:
    def __init__(self, config: dict[str, object],
                 context_window_width: int = -1):
        self._vectors = None
        self._emb_model = None
        if context_window_width <= 0:
            context_window_width = config['context_window_width']
        self.context_window_width = context_window_width
        self.config = config

    def add_context(self, row: pd.Series) -> str:
        window_start = max(0, row.start - self.context_window_width)
        window_end = min(row.end + self.context_window_width, len(row.text))
        return clean_entity(row.text[window_start:window_end])

    def _load_embeddings(self):
        self._vectors = KeyedVectors.load(self.config['keyed_vectors_file'])

    def _load_model(self):
        self._emb_model = SentenceTransformer(config['embedding_model'])

    @property
    def embeddings(self):
        if self._vectors is None:
            self._load_embeddings()
        return self._vectors

    @property
    def embedding_model(self):
        if self._emb_model is None:
            self._load_model()
        return self._emb_model

    def link(self, df: pd.DataFrame) -> list[dict]:
        mention_emb = self.embedding_model.encode(df.mention.str.lower().values)

        concepts = [self.embeddings.most_similar(m, topn=1)[0][0]
                    for m in mention_emb]
        return concepts


def highlight_text(spans: pd.DataFrame, text: str) -> list[tuple[str, object]]:
    token_concepts = [None for _ in text]

    for row in spans.itertuples():
        for k in range(row.start, row.end):
            token_concepts[k] = row.concept

    return list(zip(list(text), token_concepts))


def entity_link(query: str) -> list[tuple[str, object]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_model = AutoModelForTokenClassification.from_pretrained(
        config['segmentation_model']
    )
    seg_tokenizer = AutoTokenizer.from_pretrained(
        config['segmentation_tokenizer']
    )
    thresh = float(config['thresh'])
    query_df = pd.DataFrame({'note_id': [0], 'text': [query]})

    seg = segment(query_df, seg_model, seg_tokenizer, device, thresh)
    linked_concepts = []
    if len(seg) > 0:
        seg = seg.sort_values('start')
        linked_concepts = linker.link(seg)
    seg['concept'] = linked_concepts

    return highlight_text(seg, query)


config_parser = configparser.ConfigParser()
config_parser.read('config.ini')
config = config_parser['DEFAULT']
linker = Linker(config)

demo = gr.Interface(
    fn=entity_link,
    inputs=["text"],
    outputs=gr.HighlightedText(
        label="linking",
        combine_adjacent=True,
    ),
    theme=gr.themes.Base()
)
