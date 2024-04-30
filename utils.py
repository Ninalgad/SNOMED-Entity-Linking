import numpy as np


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
