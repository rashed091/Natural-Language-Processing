# alignment_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def pretty_print_alignments(en_words, es_words, posterior_probs):
    """
    Pretty-prints the alignment chart with Spanish words as column labels, English words as row labels, and each
    row's probabilities representing the posterior distribution over alignments for the given English word.
    :param en_words: English words
    :param es_words: Spanish words including the NULL symbol at the end
    :param posterior_probs: posterior probabilities[en idx][es idx] (so each row is a distribution)
    :return:
    """
    # probs are [en_idx][es_idx] so en is row names
    s = [[""] + es_words] + [[en_words[row_idx]] + [("%.3f" % e) for e in posterior_probs[row_idx]] for row_idx in range(0, len(posterior_probs))]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def vis_attn(source_text: List[str], target_text: List[str], activation_map: np.ndarray, filename=None, show_plot=True):
    """
    :param source_text: Source words
    :param target_text: Target words
    :param activation_map: The [src_len x trg_len] attention matrix as a numpy array
    :param filename: If not None, writes the plot to the given file (as a PDF)
    :param show_plot: True to display the plot, False to only print it
    :return:
    """
    width = 3
    word_height = 1
    pad = 0.1

    plt.figure(figsize=(4, 4))
    yoffset = 0
    xoffset = 0
    attn = activation_map
    for position, word in enumerate(source_text):
        plt.text(xoffset + pad, yoffset - (position * word_height), word, ha="right", va="center", color="k")
    for position, word in enumerate(target_text):
        plt.text(xoffset + width, yoffset - (position * word_height), word, ha="left", va="center", color="k")

    for i in range(len(source_text)):
        for j in range(len(target_text)):
            color = "b"
            plt.plot([xoffset + 2 * pad, xoffset + width - pad],
                     [yoffset - word_height * i, yoffset - word_height * j],
                     color=color, linewidth=1, alpha=attn[i, j])
    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()


def vis_attn_test():
    activation_map = np.array([[0.2, 0.8, 0.1], [0.2, 0.8, 0.1], [0.2, 0.8, 0.1], [0.2, 0.8, 0.1], [0.2, 0.8, 0.1]])
    source_text = ["this", "is", "a", "test", "!"]
    target_text = ["target", "side", "text"]
    vis_attn(source_text, target_text, activation_map, None, True)


if __name__ == "__main__":
    print("Showing a test of attention visualization -- model1_aligner.py calls this code, so you don't need to use " +
          "this main directly")
    vis_attn_test()