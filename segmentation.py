import torch.nn.functional as F
import pandas as pd

from dataloader import create_dataloader
from utils import *


def predict_segmentation(inp, model, device, batch_size=8):
    test_loader = create_dataloader(inp, batch_size)

    predictions = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = F.sigmoid(model(**batch).logits).detach().cpu().numpy()
        predictions.append(p)

    return np.concatenate(predictions, axis=0)


def create_data(text, tokenizer, seq_len=512):
    tokens = tokenizer(text, add_special_tokens=False)
    _token_batches = {k: [pad_seq(x, seq_len) for x in batch_list(v, seq_len)]
                      for (k, v) in tokens.items()}
    n_batches = len(_token_batches['input_ids'])
    return [{k: v[i] for k, v in _token_batches.items()}
            for i in range(n_batches)]


def segment_tokens(notes, model, tokenizer, device, batch_size=8):
    predictions = {}
    for note in notes.itertuples():
        note_id = note.note_id
        raw_text = note.text.lower()

        inp = create_data(raw_text, tokenizer)
        pred_probs = predict_segmentation(inp, model, device, batch_size=batch_size)
        pred_probs = np.squeeze(pred_probs, -1)
        pred_probs = np.concatenate(pred_probs)

        predictions[note_id] = pred_probs

    return predictions


def segment(notes, model, tokenizer, device, thresh, batch_size=8):
    predictions = []

    predictions_prob_map = segment_tokens(notes, model, tokenizer, device, batch_size)

    for note in notes.itertuples():

        note_id = note.note_id
        raw_text = note.text

        decoded_text = tokenizer.decode(tokenizer.encode(raw_text, add_special_tokens=False))

        pred_probs = predictions_prob_map[note_id]

        _, pred_probs = align_decoded(raw_text, decoded_text, pred_probs)
        pred_probs = np.array(pred_probs, 'float32')
        pred = (pred_probs > thresh).astype('uint8')

        spans = get_sequential_spans(pred)

        note_predictions = {'note_id': [], 'start': [], 'end': [], 'mention': [], 'score': []}
        for (start, end) in spans:
            note_predictions['note_id'].append(note_id)
            note_predictions['score'].append(pred_probs[start:end].mean())
            note_predictions['start'].append(start)
            note_predictions['end'].append(end)
            note_predictions['mention'].append(raw_text[start:end])

        note_predictions = pd.DataFrame(note_predictions)
        note_predictions = note_predictions.sort_values('score', ascending=False)

        # remove overlapping spans
        seen_spans = set()
        unseen = []
        for span in note_predictions[['start', 'end']].values:
            span = tuple(span)
            s = False
            if not is_overlap(seen_spans, span):
                seen_spans.add(span)
                s = True
            unseen.append(s)
        note_predictions = note_predictions[unseen]

        predictions.append(note_predictions)
    predictions = pd.concat(predictions).reset_index(drop=True)
    return predictions
