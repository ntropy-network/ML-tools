import math
import re
from textwrap import wrap

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


def pretty_print_ne(tokens, labels, width=80):
    if type(tokens) == str:
        tokens = tokens.split(' ')
    if type(labels) == str:
        labels = labels.split(' ')
    token_line = []
    label_line = []
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
        label_line.append(label_text)
        pad_length = max(len(token_text), len(label_text))
        token_line.append(pad_text(token_text, pad_length))

    token_line = ' '.join(token_line)
    label_line = ' '.join(label_line)
    token_lines = wrap(token_line, width=width)

    index = 0
    for i, token in enumerate(token_lines):
        end_index = index + len(token) + 1
        print(label_line[index:end_index])
        index = end_index
        print(token+'\n')


if __name__ == '__main__':
    pretty_print_ne("On April 1, 1976, Apple Computer Company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
                 "O B-DATE I-DATE I-DATE B-ORG I-ORG I-ORG O O O B-PER I-PER B-PER I-PER O B-PER I-PER")
