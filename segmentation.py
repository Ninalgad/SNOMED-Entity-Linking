import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataloader import create_dataloader


def is_overlap(existing_spans, new_span):
    for span in existing_spans:
        # Check if either end of the new span is within an existing span
        if (span[0] <= new_span[0] <= span[1]) or \
                (span[0] <= new_span[1] <= span[1]):
            return True
        # Check if the new span entirely covers an existing span
        if new_span[0] <= span[0] and new_span[1] >= span[1]:
            return True
    return False


def get_sequential_spans(a):
    spans = []

    prev = False
    start = 0

    for i, x in enumerate(a):
        if not prev and x:
            start = i
        elif prev and not x:
            spans.append((start, i))

        prev = x

    if x:
        spans.append((start, i + 1))

    return spans


def predict_segmentation(inp, model, device, batch_size=8):
    test_loader = create_dataloader(inp, batch_size)

    predictions = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = F.sigmoid(model(**batch).logits).detach().cpu().numpy()
        predictions.append(p)

    return np.concatenate(predictions, axis=0)


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def pad_seq(seq, max_len):
    n = len(seq)
    if n >= max_len:
        return seq
    else:
        return np.pad(seq, (0, max_len - n))


def create_data(text, tokenizer, seq_len=512):
    tokens = tokenizer(text, add_special_tokens=False)
    _token_batches = {k: [pad_seq(x, seq_len) for x in batch_list(v, seq_len)]
                      for (k, v) in tokens.items()}
    n_batches = len(_token_batches['input_ids'])
    return [{k: v[i] for k, v in _token_batches.items()}
            for i in range(n_batches)]


def align_decoded(x, d, y):
    clean_text = ""
    clean_label = []
    j = 0
    for i in range(len(d)):
        found = False
        for delim in [',', '.', '?', "'"]:
            if (x[j:j + 2] == f" {delim}") and (d[i] == f"{delim}"):
                found = True
                clean_text += f' {delim}'
                clean_label += [y[j], y[j]]
                j += 1

        if not found:
            clean_text += x[j]
            clean_label += [y[j]]
        j += 1

    if (clean_text != x) and (x[-1:] == "\n"):
        clean_text += "\n"
        clean_label += [0, 0]
        print('?')
    return clean_text, clean_label


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


def segment(notes, model, tokenizer, device, thresh, predictions_prob_map=None,
            batch_size=8):
    predictions = []

    if predictions_prob_map is None:
        predictions_prob_map = segment_tokens(notes, model, tokenizer, device, batch_size)

    for note in notes.itertuples():

        note_id = note.note_id
        raw_text = note.text

        decoded_text = tokenizer.decode(tokenizer.encode(raw_text, add_special_tokens=False))

        pred_probs = predictions_prob_map[note_id]

        _, pred_probs = align_decoded(raw_text, decoded_text, pred_probs)
        pred_probs = np.array(pred_probs, 'float32')
        pred = (pred_probs > thresh).astype('uint8')
        #
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
