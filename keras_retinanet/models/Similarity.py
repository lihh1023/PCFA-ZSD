from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """
    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))
    return norm

def similarity(x_y):

    # https://doi.org/10.1016/j.neucom.2021.03.073

    # x, y, word_vector, autoencoder_output=x_y

    x, y = x_y  # (w, h, 512),[1, 65, 512]
    y = K.transpose(y[0])  # [512,65]
    x_mag = l2_norm(x, axis=2)  # (w, h, 1)
    y_mag = l2_norm(y, axis=0)  # [1, 65]

    score = K.dot(x, y) / K.dot(x_mag, y_mag)  # [w,h,300]*[300,65]

    return score

def score_fusion(x_y):
    score_s2v, score_v2s = x_y
    score = 1/2 * (score_s2v + score_v2s)

    return score

def score_fusion_harmonic(x_y):
    score_s2v, score_v2s = x_y
    score = (2 * score_s2v * score_v2s) / (score_s2v + score_v2s)
    score = K.print_tensor(score, message='\n score = ')
    return score
