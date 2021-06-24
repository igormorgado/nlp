#!/usr/bin/env python

#%%
import numpy as np
from trax import layers as tl
from trax import shapes  # data signatures: dimensionality and type
from trax import fastmath  # uses jax, offers numpy on steroids

#%%
#Model = tl.Serial(
#        tl.Dense(4),
#        tl.Sigmoid(),
#        tl.Dense(4),
#        tl.Sigmoid(),
#        tl.Dense(3),
#        tl.Softmax())

#%%
# Layers
# Create a relu trax layer
relu = tl.Relu()

#%% # Inspect properties
print("-- Properties --")
print("name :", relu.name)
print("expected inputs :", relu.n_in)
print("promised outputs :", relu.n_out, "\n")

#%% Inputs
x = np.array([-2, -1, 0, 1, 2])
print("-- Inputs --")
print("x :", x, "\n")

#%%  Outputs
y = relu(x)
print("-- Outputs --")
print("y :", y)


#%%
concat = tl.Concatenate()

x1 = np.array([-10, -20, -30])
x2 = x1/ -10
y = concat([x1, x2])
print(y)

#%%
concat3 = tl.Concatenate(n_items=3)
x1 = np.array([-10, -20, -30])
x2 = x1/ -10
x3 = x1* .99
y = concat([x1, x2, x3])
print(y)

#%%
norm = tl.LayerNorm()
x = np.array([0, 1, 2, 3], dtype='float32')

norm.init(shapes.signature(x))
y=norm(x)

#%%
def TimesTwo():
    layer_name = "TimesTwo"

    def func(x):
        return x * 2

    return tl.Fn(layer_name, func)

#%%
times_two = TimesTwo()

#%%
serial = tl.Serial(
        tl.LayerNorm(),
        tl.Relu(),
        times_two,
        tl.Dense(n_units=3),
        tl.Dense(n_units=1),
        tl.LogSoftmax()
        )

#%%
x = np.array([-2, -1, 0, 1, 2] , dtype='float32')
serial.init(shapes.signature(x))

#%%
print(f'{serial.name}, {serial.sublayers}, {serial.n_in}, {serial.n_out}')
#%%
for x in range(len(serial.weights)):
    print(f'{serial.sublayers[x].name} {serial.weights[x]}')


#%%
y = serial(x)

#%%
xnp = np.array([0,1,2])
xjx = fastmath.numpy.array([0,1,2])

print(f'{type(xnp)}')
print(f'{type(xjx)}')
#%%


#%% Just an example
dense_layer = Dense(n_units=10)  #sets  number of units in dense layer
z = np.array([[2.0, 7.0, 25.0]]) # input array

dense_layer.init(z, random_key)
print("Weights are\n ",dense_layer.weights) #Returns randomly generated weights
print("Foward function output is ", dense_layer(z)) # Returns multiplied values of units and weights

#%%
tmp_embed = tl.Embedding(vocab_size=3, d_feature=2)
tmp_embed = np.array([[1,2,3],[4,5,6]])






