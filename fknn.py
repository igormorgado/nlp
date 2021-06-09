#!/usr/bin/env Python

#%%
import numpy as np
import matplotlib.pyplot as plt
from nlp_helper import *
from sentiment_utils import process_tweet

#%%
def side_of_plane(P, v):
    dotproduct = np.dot(P, v.T) # Get the dot product P * v'
    sign_of_dot_product = np.sign(dotproduct) # The sign of the elements of the dotproduct matrix
    sign_of_dot_product_scalar = sign_of_dot_product.item() # The value of the first item
    return sign_of_dot_product_scalar

#%%
def side_of_plane_matrix(P, v):
    dotproduct = np.dot(P, v.T) # Get the dot product P * v'
    sign_of_dot_product = np.sign(dotproduct) # The sign of the elements of the dotproduct matrix
    return sign_of_dot_product

#%%
def hash_multi_plane(P_l, v):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_of_plane(P,v)
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
    return hash_value

#%%
def hash_multi_plane_matrix(P, v, num_planes):
    sides_matrix = side_of_plane_matrix(P, v)
    hash_value = 0
    for i in range(num_planes):
        sign = sides_matrix[i].item()
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
    return hash_value


#%% Example

np.random.seed(0)
num_dimensions = 2
num_planes = 3
random_planes_matrix = np.random.normal(size = (num_planes, num_dimensions))

v = np.array([[2,2]])

#%%
sides_l = side_of_plane_matrix(random_planes_matrix, v)

