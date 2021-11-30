import math
import re


def pad_text(text, length):
    diff = length - len(text)
    if diff > 0:
        text += ' ' * diff
    return text


def center_text(text, length, pad_token='_'):
    diff = length - len(text)
    if diff > 0:
        padding_l = pad_token * math.floor(diff / 2)
        padding_r = pad_token * math.ceil(diff / 2)
        text = padding_l + text + padding_r
    return text


def get_label_spans(token_labels):
    out = []
    previous_label = None

    for idx, label in enumerate(token_labels):
        label_name = re.sub(r'^[BI]-', '', label)
        if re.sub(r'^[I]-', '', label) == previous_label:
            out[-1]['span_end'] += 1
        else:
            out.append({'label': label_name, 'span_start': idx, 'span_end': idx})
        previous_label = label_name
    return out


def pretty_print_ne(text, labels):
    tokens = text.split(' ')
    labels = labels.split(' ')
    line1 = []
    line2 = []
    label_spans = get_label_spans(labels)
    for label_span in label_spans:
        label_text = label_span['label']
        label_tokens = [tokens[i] for i in range(label_span['span_start'], label_span['span_end'] + 1)]
        token_text = ' '.join(label_tokens)
        pad_length = max(len(token_text), len(label_text))
        other_token = label_text == 'O'
        pad_token = ' ' if other_token else '_'
        label_text = '' if other_token else label_text
        label_text = center_text(label_text, pad_length, pad_token=pad_token)
        line1.append(label_text)
        pad_length = max(len(token_text), len(label_text))
        line2.append(pad_text(token_text, pad_length))

    print(' '.join(line1))
    print(' '.join(line2)+'\n')


def ppn(text, labels):
    return pretty_print_ne(text, labels)