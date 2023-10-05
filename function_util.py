import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    if x.max()>=5-5e-3:
        return np.ones_like(x)
    return sigmoid(x)*(1-sigmoid(x))

def tanh_prime(x):
  if x.max()>=2-5e-3:
      return np.ones_like(x)
  return 1-np.tanh(x)**2

def dropout(lstm, params, is_feed_input, d_input):
    if params.dropout<1:
        if is_feed_input: #t>=srcMaxLen && ll==1 && params.feedInput % predict words
            d_input[1:2*params.lstmSize, :] = d_input[1:2*params.lstmSize, :]*lstm.dropoutMaskInput #dropout x_t, s_t
        else:
            d_input[1:params.lstmSize, :] = d_input[1:params.lstmSize, :]*lstm.dropoutMask # dropout x_t
    #d_input(1:params.lstmSize, :) = d_input(1:params.lstmSize, :).*lstm.dropoutMask; % dropout x_t
