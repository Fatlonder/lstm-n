import numpy as np
from data_util import *
from function_util import *

row_input_dimension = 65
row_output_dimension = 65
vocab_dimension = 65
epoch = 100
lr = 1e-3
batch_size = 10
time_window = 10
null_token_index = 0
network_length = time_window
weight_matricies = 4
model = []
network_depth = 2
input_file = 'input.txt'
vocabulary_list = ['<null>__null__<null>'] # You have the index and want to retrieve the word. i.e. you use this to get preimage of embeding.
word_vocabulary = {}
init_state = {'h_t': np.random.rand(vocab_dimension, 1), 'c_t': np.random.rand(vocab_dimension, 1)}
rng = np.random.default_rng(seed=42)


def time_backprop(model, lstm):
    grads = [0] * time_window
    dc = np.zeros((vocab_dimension, 1)) # Get it from somewhere!
    for t in range(time_window-1, -1, -1):
        delta = grads[t+1][1] if t<time_window-1 else 1
        dh = lstm[t][network_depth-1]['dh'] # *delta. Passing the gradient from prev time step. 
        grads[t] = layer_backprop(model[t], lstm[t], dc, dh)
    return grads


def layer_backprop(layer_w, current_state, dc, dh):
    grad_w = np.zeros((network_depth, weight_matricies, row_output_dimension, row_input_dimension))
    c_t_1 = current_state[network_depth-1]['c_t_1']
    for i in reversed(range(network_depth)):
        dc, dh, grad_w[i] = cell_backprop(layer_w[i], current_state[i], dc, dh, c_t_1)
        c_t_1 = current_state[i-1]['c_t_1']
    return (grad_w, dh)


def cell_backprop(w, lstm, dc, dh, c_t_1):
    """
    Gradient of a cell 
    wrt to c_t, h_t, input (x_{t}, h_{t-1}), 
    and weights matrix(es) w^{i}, w^{f}, w^{o}, w^{a}
    """
    d_i_in = np.multiply(lstm['a_signal'], sigmoid_prime(lstm['i_in']))
    d_a_in = np.multiply(tanh_prime(lstm['a_in']), lstm['i_gate'])
    d_f_in = np.multiply(sigmoid_prime(lstm['f_in']), c_t_1)
    dh_ct = np.multiply(tanh_prime(lstm['c_t']), lstm['c_t'])
    
    dc = dc + np.multiply(dh_ct, lstm['o_gate'])*dh # Add dc from the cell above, in order to accomulate the change needed to lower the loss. 
    di = np.multiply(dc, d_i_in)
    df = np.multiply(dc, d_f_in)
    do = np.multiply(lstm['t_c_t'], sigmoid_prime(lstm['o_in']))*dh
    da = np.multiply(dc, d_a_in) 
    i_f_o_a = [di, df, do, da]

    input_m = np.full_like(np.zeros((row_input_dimension, row_output_dimension)), lstm['input'])
    di_m = np.full_like(np.zeros((di.shape[0], di.shape[0])), di).T
    df_m = np.full_like(np.zeros((df.shape[0], df.shape[0])), df).T
    do_m = np.full_like(np.zeros((do.shape[0], do.shape[0])), do).T
    da_m = np.full_like(np.zeros((da.shape[0], da.shape[0])), da).T
    i_f_o_a_m = [di_m, df_m, do_m, da_m]

    d_w = np.multiply(i_f_o_a_m, input_m) # Shape -> (4, dim(W))
    d_prev_h = w@i_f_o_a # Shape -> (4, dim(dx)), x \in {i, f, o, a}
    d_f_i_a = (d_prev_h[1]) +  d_prev_h[3] + d_prev_h[0] # (w['f']@df) +  w['a']@da + w['i']@di = fg' + gf', wrt h_{t-1}
    dh = np.multiply(np.tanh(lstm['c_t']), d_prev_h[2]) + np.multiply(lstm['o_gate'], d_f_i_a) # fg' + gf'.
    dc = np.multiply(dc, lstm['f_gate'])
    return (dc, dh, d_w)

def time_forward(model, init_state, time_context_length, training_data):
    input, target = training_data # No batches now, will do it latter!
    lstm_states = []
    loss = 0
    for t in range(time_context_length):
        prev_state = lstm_states[t-1][network_depth-1] if t>=1 else init_state
        input_emb = np.zeros((vocab_dimension, 1))
        input_emb[input[t]] = 1
        lstm_states.append(layer_forward(model[t], prev_state, input_emb))
        y_t = lstm_states[t][network_depth-1]['h_t']
        e_y_t_norm, e_y_t = np.sum(np.exp(y_t)), np.exp(y_t) 
        p_t = e_y_t/e_y_t_norm # ...and normalize the vector.
        loss += -np.log(p_t[target[t]]) # The surprise that we want to lower. 
        """
        calculating d_h = \frac{\partial L(p_t)}{\partial h}.
        """
        e_y_t_m = np.array(e_y_t.tolist()*vocab_dimension).reshape((vocab_dimension, vocab_dimension)).T # Each element per corresponding row has the same value
        e_y_t_mdiag = np.array(e_y_t.tolist()*vocab_dimension).reshape((vocab_dimension, vocab_dimension)).T
        np.fill_diagonal(e_y_t_mdiag, 0)
        d_pi = np.diag(e_y_t_m@e_y_t_mdiag)/e_y_t_norm**2
        d_h = -1/p_t[target[t]]*d_pi
        lstm_states[t][network_depth-1]['dh'] = d_h[t]
        """
        nats_vector = -np.log(prob) # nat (units) of information
        nats_vector_norm = np.sqrt(np.dot(nats_vector, prob))
        target_norm = training_data[t] # The target_vector = [0, 0, ..., number, 0, ..., 0] then the norm is simply the 'number'. 
        prob_inner_target = np.dot(nats_vector, target_norm)
        cos_similarity = prob_inner_target/(nats_vector_norm*prob_inner_target)
        loss+= cos_similarity # want to lower the nat (units) of information ("surprise")
        """
    return lstm_states, loss

def layer_forward(model, prev_time_state, input_emb):
    lstm_state = []
    for l in range(network_depth):
        h_t_1 = prev_time_state['h_t'] if l<1 else lstm_state[l-1]['h_t'] 
        c_t_1 = prev_time_state['c_t'] if l<1 else lstm_state[l-1]['c_t']
        x_t = lstm_state[l-1]['h_t'] if l>=1 else input_emb
        input_state = (x_t, h_t_1)
        lstm_state.append(cell_forward(model[l], input_state, c_t_1))
    return lstm_state

def cell_forward(w, input_state, c_t_1):
    cell_state = {}
    ifoa_linear = w@(input_state[0]+input_state[1]) # w = [w_i, w_f, w_o, w_a]
    i_in, f_in, o_in, a_in = ifoa_linear[0, :], ifoa_linear[1, :], ifoa_linear[2, :], ifoa_linear[3, :]
    i_gate, f_gate, o_gate, a_signal = sigmoid(i_in), sigmoid(f_in), sigmoid(o_in), np.tanh(a_in)
    c_t = np.multiply(f_gate, c_t_1) + np.multiply(i_gate, a_signal)
    t_c_t = np.tanh(c_t)
    h_t = np.multiply(t_c_t, o_gate)

    cell_state['h_t'] = h_t
    cell_state['c_t'] = c_t
    cell_state['c_t_1'] = c_t_1
    cell_state['t_c_t'] = t_c_t
    cell_state['input'] = input_state[1]
    cell_state['i_gate'] = i_gate
    cell_state['f_gate'] = f_gate
    cell_state['o_gate'] = o_gate
    cell_state['a_signal'] = a_signal
    cell_state['a_in'] = a_in
    cell_state['o_in'] = o_in
    cell_state['i_in'] = i_in
    cell_state['f_in'] = f_in
    return cell_state

def train_lstm(model, training_data, time_window):
    loss = 0
    for i in range(len(training_data)):
        lstm, loss = time_forward(model, init_state, time_window, training_data[i])
        if i %100 ==0:
            out_value = lstm[time_window-1][network_depth-1]['h_t']
            #print(f"The output is: ({out_value}), with loss: {loss}")
        gradients = time_backprop(model, lstm)
        model = update_weights(model, gradients, lr)
        lstm = update_lstm(lstm, gradients[0][1][0], lr)

    return (loss, model)

def update_weights(model, gradient, lr):
    for t in range(time_window): 
        model[t] += gradient[t][0]*lr
    return model

def update_lstm(lstm, dh, lr):
    # This is for when we cold start the network. 
    # I will pass dc latter. 
    init_state['h_t'] += dh*lr
    init_state['c_t'] += dh*lr
    # This is while training. If we store the network then we don't need init_state at all. 
    lstm[0][0]['h_t'] += dh*lr
    lstm[0][0]['c_t'] += dh*lr
    return lstm


if __name__ == "__main__":
    data = load_data(input_file, vocabulary_list)
    training_data = batch_training_data(data, batch_size, time_window, null_token_index)
    model = init_model(network_length, network_depth, weight_matricies, row_output_dimension, row_input_dimension, rng)

    for i in range(epoch):
        loss, model = train_lstm(model, training_data, time_window)
        print(f"On epoch ({i}) the loss is: ({loss})")