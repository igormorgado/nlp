#!/usr/bin/env Python

import numpy as np
import pandas as pd

#%%
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    #cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos

#%%
def build_dict(filename):
    """Read a csv with two fields and return as a dictionary"""
    df = pd.read_csv(filename, delimiter=' ', header=None, index_col=0)
    # A dict here isn't the best data structure since it collapses
    # words mapping to two different words.
    # Maybe: list(zip(df[0], df[1]))
    # Or a list inside the dict, for a bucket.
    # Also will probably need anotheer dict in the backwards direction
    return df[1].to_dict()

#%%
def pickone(embeddings):
    """Randomly return a embedded vector entry"""
    return np.random.choice(list(embeddings.keys()))

#%%
import matplotlib.pyplot as plt
def plot_vectors(vectors, colors=['k', 'b', 'r', 'm', 'c'], axes=None, fname='image.svg', ax=None):
    """ Procedure to plot and arrows that represents vectors with pyplot"""
    scale = 1
    scale_units = 'x'
    x_dir = []
    y_dir = []

    for i, vec in enumerate(vectors):
        x_dir.append(vec[0][0])
        y_dir.append(vec[0][1])

    if ax == None:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax

    if axes == None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]

    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])

    arrows = []
    for i, vec in enumerate(vectors):
        arrows.append(ax2.arrow(0, 0, vec[0][0], vec[0][1], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i]))

    if ax == None:
        plt.show()
        fig.savefig(fname)
    else:
        return arrows


