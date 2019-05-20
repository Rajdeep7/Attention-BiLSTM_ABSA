import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.rnn import LSTMStateTuple
import tensorflow.contrib.layers as layers


def get_a_cell(num_units, dropout = False):
    cell = LSTMCell(num_units)
    if dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.8)
    return cell


# Defined cell entries as functions
def cell_maker(num_units = 64, num_of_stacks = 1):
    # cell = BNLSTMCell(80, is_training)
    if num_of_stacks == 1:
        # if 1 layer, return cells themselves
        return LSTMCell(num_units)
    else:
        # if multiple layers, wrap cell lists into MultiRNNCell's
        cells = [get_a_cell(num_units, dropout = False) for _ in range(num_of_stacks)]
        return MultiRNNCell(cells)


def fuse_aspect_words(aspect_embeddings, args):
    """
    This method fuses aspect word embeddings to generate a fixed sized aspect embedding

    :param aspect_embeddings: [batch_size, max_aspect_words_count, aspect_embedding_size]
    :param args:
    :return: compressed_aspect_embedding [batch_size, aspect_embedding_size]
    """
    if args.aspect_fusion == 'mean':
        compressed_aspect_embedding = tf.reduce_mean(aspect_embeddings, axis = 1)
    elif args.aspect_fusion == 'max':
        compressed_aspect_embedding = tf.reduce_max(aspect_embeddings, axis = 1)
    elif args.aspect_fusion == 'projection':
        aspect_projection = layers.fully_connected(aspect_embeddings, args.aspect_embedding_size)
        compressed_aspect_embedding = tf.reduce_mean(aspect_projection, axis = 1)
    else:
        compressed_aspect_embedding = tf.reduce_mean(aspect_embeddings, axis = 1)

    return compressed_aspect_embedding


def bidirectional_rnn(cell_fw,
                      cell_bw,
                      inputs_embedded,
                      input_lengths,
                      scope = None):
    """
    Bidirecional RNN with concatenated outputs and states
    https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
    https://damienpontifex.com/2017/12/06/understanding-tensorflows-rnn-inputs-outputs-and-shapes/

    The initial state for both directions is zero by default (but can be set optionally) and no intermediate states
    are ever returned -- the network is fully unrolled for the given (passed in) length(s) of the sequence(s) or
    completely unrolled if length(s) is not given.

    fw_outputts = [batch_size, max_time, num_units] : this is h(t) output for every timestep.
    output_states: A tuple (output_state_fw, output_state_bw) containing the forward and the backward
    final states of bidirectional rnn. This is the final state of the cell at the end of the sequence
    fw_state = [batch_size, num_units]
    """
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw,
                                                                                           cell_bw = cell_bw,
                                                                                           inputs = inputs_embedded,
                                                                                           sequence_length = input_lengths,
                                                                                           dtype = tf.float32,
                                                                                           swap_memory = True,
                                                                                           scope = scope)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat((fw_state.c, bw_state.c), 1, name = 'bidirectional_concat_c')
                state_h = tf.concat((fw_state.h, bw_state.h), 1, name = 'bidirectional_concat_h')
                state = LSTMStateTuple(c = state_c, h = state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1, name = 'bidirectional_concat')
                return state
            elif isinstance(fw_state, tuple) and isinstance(bw_state, tuple) and len(fw_state) == len(bw_state):
                # multilayer
                state = tuple(concatenate_state(fw, bw) for fw, bw in zip(fw_state, bw_state))
                return state
            else:
                raise ValueError('unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)
        outputs = tf.concat((fw_outputs, bw_outputs), 2, name='birnn_output')

        return outputs, state


def task_specific_attention(inputs,
                            output_size,
                            args,
                            initializer = layers.xavier_initializer(),
                            activation_fn = tf.tanh,
                            scope = None,
                            sequence_length = None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within a task of interest).

    :param inputs: Tensor of shape [batch_size, units, input_size]
                    `input_size` must be static (known)
                    `units` axis will be attended over (reduced to output)
                    `batch_size` will be preserved
    :param output_size: Size of the output attention vector.
    :param initializer:
    :param activation_fn:
    :param scope:
    :return: outputs: Tensor of shape [batch_size, output_size].
    """

    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        """
        It is a learnable vector which will learn and have information that which units(words/sentences)are worth paying
        attention too.
        """
        attention_context_vector = tf.get_variable(name = 'attention_context_vector',
                                                   shape = [output_size],
                                                   initializer = initializer,
                                                   dtype = tf.float32)

        """
        Here, we feed LSTM word annotations h_it through a one layer MLP to get u_it as a hidden representation
        of h_it. This is specific to HAN paper.
        input_projection = [batch_size, max_time, output_size]
        """
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn = activation_fn,
                                                  scope = scope)

        """
        Similarity measure/score is computed between context vector and embeddings using
        tf.multiply(input_projection, attention_context_vector)
        vector_attn = [batch_size, max_time, 1]
        """
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis = 2, keep_dims = True)

        """
        Attention weights are computed by applying softmax on similarity scores
        attegntion_weights = [batch_size, max_time, 1]
        """
        attention_weights = tf.nn.softmax(vector_attn, dim = 1)

        """
        outputs = [batch_size, output_size] : It is the weighted sum of attention_weights with units(word embeddings).
        Originally in paper h_it are used but here in this code we are using u_it.
        weighted_projection = [batch_size, max_time, output_size]
        """
        if args.use_combined_attention_input:
            weighted_projection = tf.multiply(input_projection, attention_weights)
        else:
            size = 2 * args.word_cell_units
            weighted_projection = tf.multiply(inputs[:, :, 0:size], attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis = 1)
        # outputs = tf.reduce_mean(weighted_projection, axis = 1)

        return outputs
