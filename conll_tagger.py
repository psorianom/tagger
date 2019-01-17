#!/usr/bin/env python
import glob
import os
import time
import codecs
import optparse
import numpy as np
from utils import create_input, iobes_iob, iob_ranges, zero_digits
from model import Model

from loader import cap_feature

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="",
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)
optparser.add_option(
    "--outputFormat", default="",
    help="Output file format"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
# assert os.path.isfile(opts.input)

# Load existing model
print("Loading model...")
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

# f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()
lower = parameters['lower']
zeros = parameters['zeros']
print('Tagging...')


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
        })
    return data

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, input_file_path):
    """
    Evaluate current model using CoNLL script.
    """
    predictions = []
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]

        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)

        for i, y_pred in enumerate(y_preds):
            new_line = "%s %s" % (raw_sentence[i][0], p_tags[i])
            predictions.append(new_line)
        predictions.append("")
    output_path = os.path.join(opts.output, os.path.basename(input_file_path[:-4] + "_Tagged.txt"))
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))

paths_input = []
if os.path.isdir(opts.input):
    for file_input in glob.glob(os.path.join(opts.input, "*.txt"))[:100]:
        paths_input.append(file_input)
else:
    paths_input.append(opts.input)

for file_path in paths_input:
    print("Tagging {}...".format(file_path))
    test_sentences = load_sentences(file_path, zeros=zeros)
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)
    evaluate(parameters, f_eval, test_sentences, test_data, model.id_to_tag, file_path)

