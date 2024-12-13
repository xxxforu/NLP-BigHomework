# 加载模型权重
from keras.models import load_model
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')

# 推理模型搭建
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # this target_seq you can treat as initial state

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        # argmax: Returns the indices of the maximum values along an axis
        # just like find the most possible char
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # find char using index
        sampled_char = reverse_target_char_index[sampled_token_index]
        # and append sentence
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        # append then ?
        # creating another new target_seq
        # and this time assume sampled_token_index to 1.0
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        # update states, frome the front parts
        states_value = [h, c]

    return decoded_sentence

# 进行模型推理

for seq_index in range(100,200):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


