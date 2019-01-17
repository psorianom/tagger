#!/usr/bin/env python
# ../../../models/tag_scheme=iob,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=100,word_lstm_dim=100,word_bidirect=True,pre_emb=jurinet_parsed_100.vec,all_emb=False,cap_dim=1,crf=True,dropout=0.5,lr_method=sgd-lr_.005
import codecs
import os
import time
import optparse

import numpy as np
from loader import load_sentences
from model import Model

from loader import prepare_dataset
from utils import evaluate
from utils import create_input, iobes_iob


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


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
assert os.path.isfile(opts.input)


# Load existing model
print
"Loading model..."
model = Model(model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]
model.id_to_tag
# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()

# f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()

def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, input_path, output_path):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    y_trues_tags = []
    y_preds_tags = []
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]

        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        y_trues_tags.extend(p_tags)
        y_preds_tags.extend(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(output_path, "%s.output" % os.path.basename(input_path))
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print(line)

    # Remove temp files
    # os.remove(output_path)
    # os.remove(scores_path)

    # Confusion matrix with accuracy for each tag
    confusion_matrix_head = "{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)
    confusion_matrix_head = confusion_matrix_head.format("ID", "NE", "Total",
                                                         *([id_to_tag[i] for i in range(n_tags)] + ["Percent"]))
    print(confusion_matrix_head)
    for i in range(n_tags):
        confusion_matrix_content = "{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)
        confusion_matrix_content = confusion_matrix_content.format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in range(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))]))
        print(confusion_matrix_content)
    print()
    print("Global accuracy")
    print("\t%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))

    # F1 on all entities
    print("F1 on all entities")
    F1_all = float(eval_lines[1].strip().split()[-1])
    print("\t{}".format(F1_all))
    return F1_all



'Tagging...'
lower = parameters['lower']
zeros = parameters['zeros']
test_sentences = load_sentences(opts.input, zeros=zeros)
test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)
test_score = evaluate(parameters, f_eval, test_sentences, test_data, model.id_to_tag,
                      opts.input, opts.output)

# f_output.close()
