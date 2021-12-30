# model1_aligner.py

import argparse
import numpy as np
import scipy.special
import re
import time
from collections import Counter
from typing import List
from utils import *

from alignment_visualizer import *


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='model1_aligner.py')
    parser.add_argument('--es', type=str, default='data/europarl-short.es-en.es', help='Spanish side of the data')
    parser.add_argument('--en', type=str, default='data/europarl-short.es-en.en', help='English side of the data')
    parser.add_argument('--max_lines', type=int, default=10000, help='max number of lines to read from the data')
    parser.add_argument('--em_iters', type=int, default=10, help='number of EM iterations to run')
    parser.add_argument('--unk_threshold', type=int, default=2.0, help='unk threshold')
    parser.add_argument('--show_plot', dest='show_plot', default=False, action='store_true', help='show the plots in addition to writing to a file')
    parser.add_argument('--make_vis', dest='make_vis', default=False, action='store_true', help='make visualizations at all')
    args = parser.parse_args()
    return args


def tokenize(string: str) -> List[str]:
    """
    Simple English tokenizer for a given string. We apply this to Spanish as well even though it's not designed
    for that purpose.
    :param string:
    :return:
    """
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\"", " \" ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return list(filter(lambda x: len(x) > 0, string.split(" ")))


def read_bitext(es_file: str, en_file: str, max_lines=10000, max_length=15) -> (List[List[str]], List[List[str]]):
    """
    Reads the given Spanish and English files, visiting the first max_lines sentences and keeping those with a Spanish
    size less than or equal to max_length after tokeniation
    :param es_file: Spanish sentences
    :param en_file: English sentences
    :param max_lines: The max number of lines to consider as training data
    :param max_length: The max length of the sentences to align (drop any sentence Spanish length greater than this)
    :return: The text in the given file as a single string
    """
    es_lines = []
    en_lines = []
    es_reader = open(es_file, "r", encoding="utf-8")
    en_reader = open(en_file, "r", encoding="utf-8")
    es_line = es_reader.readline().rstrip()
    en_line = en_reader.readline().rstrip()
    i = 0
    while i < max_lines:
        es_line_split = [word.lower() for word in tokenize(es_line)]
        en_line_split = [word.lower() for word in tokenize(en_line)]
        # print(repr(es_line_split) + "      " + repr(len(es_line_split)))
        if 0 < len(es_line_split) <= max_length and 0 < len(en_line_split):
            es_lines.append(es_line_split)
            en_lines.append(en_line_split)
            # print("============")
            # print(repr(es_line_split))
            # print(repr(en_line_split))
        es_line = es_reader.readline().rstrip()
        en_line = en_reader.readline().rstrip()
        i += 1
    print("Read %i lines of length %i or less out of %i possible" % (len(es_lines), max_length, max_lines))
    return (es_lines, en_lines)


def build_vocab(lines: List[List[str]], unk_threshold = 2.0) -> Counter:
    """
    Builds a vocabulary from the given tokenized sentences
    :param lines:
    :param unk_threshold:
    :return:
    """
    counts = Counter()
    for line in lines:
        for word in line:
            counts[word] += 1.0
    pruned_counts = Counter({k: counts[k] for k in counts if counts[k] > unk_threshold})
    print("Kept %i words out of %i possible" % (len(pruned_counts), len(counts)))
    return pruned_counts


def infer_model_1(es_words_indexed: List[int], en_words_indexed: List[int], es_en_log_probs: np.ndarray) -> np.ndarray:
    """
    Does inference in IBM Model 1 to compute P(a_i|w^es,w^en) for each word index i in the English sentence.
    :param es_words_indexed: Spanish word indices, including a NULL symbol at the end
    :param en_words_indexed: English word indices
    :param es_en_log_probs: Model 1 parameters: the log probability matrix P(w^en|w^es)
    :return: A numpy array where each row is the posterior distribution over alignment choices for each English word
    """
    probs_each_word = []
    for i in range(0, len(en_words_indexed)):
        sent_scores = [es_en_log_probs[es_words_indexed[j]][en_words_indexed[i]] for j in range(0, len(es_words_indexed))]
        probs_each_word.append(scipy.special.softmax(sent_scores))
    return np.stack(probs_each_word)


def train_model_1(es_lines: List[List[str]], en_lines: List[List[str]], args):
    """
    Learns IBM Model 1 parameter weights on the given Spanish-English bitext.
    :param es_lines: list of tokenized Spanish sentences
    :param en_lines: list of tokenized English sentences parallel with the Spanish sentences
    :param args: command-line args bundle
    :return:
    """
    null_alignment_token = "NULL"
    es_vocab = build_vocab(es_lines, unk_threshold=args.unk_threshold)
    en_vocab = build_vocab(en_lines, unk_threshold=args.unk_threshold)
    print("Spanish and English vocabularies by frequency:")
    print(repr(es_vocab))
    print(repr(en_vocab))
    es_indexer = Indexer()
    es_indexer.add_and_get_index(null_alignment_token)
    for word in es_vocab:
        es_indexer.add_and_get_index(word)
    en_indexer = Indexer()
    en_indexer.add_and_get_index(null_alignment_token)
    for word in en_vocab:
        en_indexer.add_and_get_index(word)
    # Distribution over en words for each es word P(a_i) P(en_i|es_{a_i}). So we are aligning en to es
    es_en_log_probs = None
    print("Running EM for %i iters" % args.em_iters)
    for t in range(0, args.em_iters):
        # E-step
        print("Starting E-step %i" % t)
        # Initialize count matrix to be not-quite-zero to avoid getting -Inf log probabilities
        es_en_counts = np.ones([len(es_indexer), len(en_indexer)], dtype=float) * 1e-8
        for ex_idx in range(len(es_lines)):
            # Add NULL token to the end
            es_words_indexed = [es_indexer.index_of(es_lines[ex_idx][k]) for k in range(0, len(es_lines[ex_idx]))] + [es_indexer.index_of(null_alignment_token)]
            en_words_indexed = [en_indexer.index_of(en_lines[ex_idx][k]) for k in range(0, len(en_lines[ex_idx]))]
            if es_en_log_probs is not None:
                posterior_probs = infer_model_1(es_words_indexed, en_words_indexed, es_en_log_probs)
            else:
                # In the first E-step, we assume a uniform posterior.
                posterior_probs = np.asarray([[1.0/len(es_words_indexed) for wes in es_words_indexed] for wen in en_words_indexed])
            # Count up "fractional counts" given by the posterior probabilities. This is like counting up transitions
            # and emissions in supervised HMM parameter estimation, but now we're using the model's own posterior outputs
            # as "soft labels" for the data.
            for i in range(0, len(en_words_indexed)):
                for j in range(0, len(es_words_indexed)):
                    es_en_counts[es_words_indexed[j]][en_words_indexed[i]] += posterior_probs[i][j]
        # M-step: normalize the counts accumulated so far
        print("Starting M-step %i" % t)
        normalizer = np.expand_dims(np.log(np.sum(es_en_counts, axis=1)), axis=1)
        es_en_log_probs = np.log(es_en_counts) - normalizer
        # print(repr(es_en_log_probs))
    # Print the parameters
    # Display the highest log-probability English words conditioned on each Spanish word
    print("Printing a few of the Model 1 parameter values:")
    for i in range(0, min(20, len(es_indexer))):
        counter = Counter()
        for en_word in range(0, len(en_indexer)):
            counter[en_indexer.get_object(en_word)] = es_en_log_probs[i][en_word]
        print("Highest words in the distribution P(w_en|w_es=" + repr(es_indexer.get_object(i)) + "): " + repr(counter.most_common(5)))
    # Draw 20 alignment graphs and print the alignment probabilities
    for i in range(0, min(100, len(es_lines))):
        es_words_indexed = [es_indexer.index_of(es_lines[i][k]) for k in range(0, len(es_lines[i]))] + [es_indexer.index_of(null_alignment_token)]
        en_words_indexed = [en_indexer.index_of(en_lines[i][k]) for k in range(0, len(en_lines[i]))]
        posterior_probs = infer_model_1(es_words_indexed, en_words_indexed, es_en_log_probs)
        pretty_print_alignments(en_lines[i], es_lines[i] + [null_alignment_token], posterior_probs)
        # Alignments are from en to es
        if i < 10 and args.make_vis:
            vis_attn(en_lines[i], es_lines[i] + [null_alignment_token], posterior_probs, filename="test-%i.pdf" % i, show_plot=args.show_plot)


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    (es_lines, en_lines) = read_bitext(args.es, args.en, max_lines=args.max_lines)
    train_model_1(es_lines, en_lines, args)
