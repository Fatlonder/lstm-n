from lstm import *


def test_batch_loading():
    data = load_data(input_file, [])
    chars = list(set(data))
    training_data = batch_training_data(data, batch_size, time_window, null_token_index)
    print(training_data)

def test_dummy_data_load(input_file, time_window):
    return dummy_load_data(input_file, time_window)

def test_train_lstm():
    training_data = dummy_load_data(input_file, time_window)
    model = init_model(network_length, network_depth, weight_matricies, row_output_dimension, row_input_dimension, rng)
    for i in range(epoch):
        loss, model = train_lstm(model, training_data, time_window)
        print(f"On epoch ({i}) the loss is: ({loss}), Model Max/Min is : {model.max()}/{model.min()}")

def test_time_backward(model, init_state):
    time_backprop(model, init_state)
    
if __name__ =="__main__":
    #data = test_dummy_data_load('input.txt', 10)
    #test_train_lstm()
    #test_time_backward(model, init_state)
    test_train_lstm()