#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 06:28:46 2019
Forward Pass of a Neural Network
Source/Reference:CSE6040 Solutions 
"""

import numpy as np
import time 


#estimate time of the baseline implementation 
def rel_error(x, y):
    return np.max(np.abs(x-y)/
                  (np.maximum(1e-8, np.abs(x)+np.abs(y))))



#fully connected layer for images 
def FC_naive(x ,w, b):
    # x = numpy array of images of shape(N,H,W)
    # w = numpy array of weights of shape (M,H,W)
    # b = numpy vector of biases of size M 
    N,H,W = x.shape
    M,_,_ = w.shape
    out = np.zeros((N, M)) #output
    for ni in range(N):
        for mi in range(M):
            out[ni,mi] = b[mi]    
            for d1 in range(H):
                for d2 in range(W):
                    out[ni,mi] += x[ni, d1, d2] * w[mi,d1,d2]
    return out 


number_inputs = 50
input_shape = (128, 256)
output_dimension = 10
                  
x = np.random.rand(number_inputs, *input_shape) 
w = np.random.rand(output_dimension, *input_shape)
b = np.random.rand(output_dimension)
start_time = time.time()
out = FC_naive(x,w,b)
elapsed_time = time.time() - start_time
print("Took %g seconds." % elapsed_time)


 #rewrite function and replace two inner loops
def two_inner_loops(x_i, w_l, b_l):
    return np.sum(np.multiply(x_i, w_l)) + b_l 
    
    
#fc naive function using two inner loops
def FC_two_loops(x, w, b):
    N, H, W = x.shape
    M, _, _ = w.shape
    out = np.zeros((N, M)) 
    for ni in range(N):
        for mi in range(M):
            out[ni, mi] = two_inner_loops(x[ni, :, :], w[mi, :, :], b[mi])
    return out

number_inputs = 50 
input_shape = (128, 256)
output_dimension = 10 

x = np.random.rand(number_inputs, *input_shape) 
w = np.random.rand(output_dimension, *input_shape)
b = np.random.rand(output_dimension)

start_time = time.time()
for i in range(5):
    out_fast = FC_two_loops(x, w, b)
    
elapsed_time_fast = (time.time()-start_time)/5
print("Took %g seconds." % elapsed_time_fast)

start_time = time.time()
for i in range(5):
    out_naive = FC_naive(x, w, b)
    
elapsed_time_naive = (time.time()-start_time)/5
print("Took %g seconds." % elapsed_time_naive)

error = rel_error(out_naive, out_fast)
print("Output error:", error)

speed_up = elapsed_time_naive/elapsed_time_fast
print("Speed up:", speed_up)


#new function for fc naive 
def FC_no_loop(x, w, b):
    N, H, W = x.shape
    M, _, _ = w.shape
    out = np.zeros((N,M))
    x = np.reshape(x, (N, H*W))
    w = np.reshape(w, (M, H*W))
    out = x @ w.T + b 
    return out 

number_inputs = 50 
input_shape = (128, 256)
output_dimension = 10 

x = np.random.rand(number_inputs, *input_shape) 
w = np.random.rand(output_dimension, *input_shape)
b = np.random.rand(output_dimension)

start_time = time.time()
for i in range(5):
    out_fast = FC_two_loops(x, w, b)
    
elapsed_time_fast = (time.time()-start_time)/5
print("FC_no_loop took %g seconds." % elapsed_time_fast)

start_time = time.time()
for i in range(5):
    out_naive = FC_naive(x, w, b)
    
elapsed_time_naive = (time.time()-start_time)/5
print("FC_no_loop took %g seconds." % elapsed_time_naive)

error = rel_error(out_naive, out_fast)
print("FC_no_loop Output error:", error)

speed_up = elapsed_time_naive/elapsed_time_fast
print("FC_no_loop Speed up:", speed_up)

