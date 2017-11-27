import tensorflow as tf


class MixAndGenerateProbabilities:
    """ This class mixes embeddings of text and image
     and generates probabilities over action space.
     Specifically, given a,b as embedding of text and image
     respectively. It uses 2 layer MLP with relu-units to generate
     first an embedding c and then uses c to generate probabilties
     over block IDs and 4 actions.
    """

    def __init__(self, n_text, n_image, n_previous_action, text_embed, image_embed, previous_action_embed,
                 num_actions, use_softmax=True, scope_name="mix", create_copy=None):

        batchsize = tf.shape(image_embed)[0]

        # Compute attention vector
        with tf.variable_scope("attention"):

            if create_copy is not None:
                tf.get_variable_scope().reuse_variables()

            #pre-process keys
            attention_vectorized = tf.reshape(image_embed, [batchsize, 225, 32])
            weights = tf.get_variable('weights', [32, 32], initializer=tf.truncated_normal_initializer(stddev=0.004))

            batched = tf.reshape(attention_vectorized,[batchsize*225,32])
            batched_keys = tf.matmul(batched, weights)
            keys = tf.reshape(batched_keys,[batchsize,225,32])

            #pre-process query
            weights_2 = tf.get_variable('weights_2', [250, 32], initializer=tf.truncated_normal_initializer(stddev=0.004))
            biases_2 = tf.get_variable('biases_2', [32], initializer=tf.constant_initializer(0.0))

            text_embed_real = tf.reshape(text_embed, [batchsize, 1, 250])
            batched_2 = tf.reshape(text_embed_real, [batchsize * 1, 250])
            batched_queries = tf.matmul(batched_2, weights_2)
            queries = tf.reshape(batched_queries, [batchsize, 1, 32])
            queries = tf.add(queries, biases_2, "attention")

            print(queries)

            pre_tanh = tf.add(keys, queries)
            print(pre_tanh)
            keys_queries = tf.tanh(pre_tanh)

            weights_3 = tf.get_variable('weights_new', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.004))
            biases_3 = tf.get_variable('biases_new', [32], initializer=tf.constant_initializer(0.0))

            batched_3 = tf.reshape(keys_queries, [batchsize * 225, 32])
            batched_final = tf.matmul(batched, weights)
            attention_pre = tf.reshape(batched_final, [batchsize, 225, 1])

            finalAttention = tf.nn.softmax(tf.add(attention_pre, biases_3, "attention"))

            print("finally")
            print(finalAttention)

            #APPLY THE thing

            image_embed_weighted = tf.multiply(attention_vectorized, finalAttention)
            image_embed = tf.reshape(image_embed_weighted, [batchsize, 15, 15, 32])

            # FINISH IMAGE PROCESSING?
            with tf.variable_scope("linearImage") as scope:
                if create_copy is not None:
                    tf.get_variable_scope().reuse_variables()
                # Move everything into depth so we can perform a single matrix multiply.
                reshape = tf.reshape(image_embed, [batchsize, -1])
                # Value before is hacked
                # Not sure how to fix it
                # It if based on image dimension
                dim = 160
                weights = tf.get_variable('weights', [dim, 200],
                                          initializer=tf.truncated_normal_initializer(stddev=0.004))
                biases = tf.get_variable('biases', [200],
                                         initializer=tf.constant_initializer(0.0))
                image_embed = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)

        observed_state = tf.concat(1, [image_embed, text_embed, previous_action_embed])
        n_input = n_image + n_text + n_previous_action
        self.n_actions = num_actions
        dim = 120
        n_block = 20

        if create_copy is not None:
            self.weights = create_copy.weights
            self.biases = create_copy.biases
        else:
            with tf.name_scope(scope_name):
                # layers weight & bias
                self.weights = {
                    'w_1': tf.Variable(tf.random_normal([n_input, dim], stddev=0.01)),
                    'w_dir': tf.Variable(tf.random_normal([dim, self.n_actions], stddev=0.01)),
                    'w_block': tf.Variable(tf.random_normal([dim, n_block], stddev=0.01))
                }
                self.biases = {
                    'b_1': tf.Variable(tf.constant(0.0, dtype=None, shape=[dim])),
                    'b_dir': tf.Variable(tf.constant(0.0, dtype=None, shape=[self.n_actions])),
                    'b_block': tf.Variable(tf.constant(0.0, dtype=None, shape=[n_block]))
                }

        # Compute a common representation
        layer = tf.nn.relu(tf.add(tf.matmul(observed_state, self.weights["w_1"]), self.biases["b_1"]))

        # Direction logits 
        direction_logits = tf.add(tf.matmul(layer, self.weights["w_dir"]), self.biases["b_dir"])
        
        # Block logits
        block_logits = tf.add(tf.matmul(layer, self.weights["w_block"]), self.biases["b_block"])
        
        if use_softmax:
            self.direction_prob = tf.nn.softmax(direction_logits)
            self.block_prob = tf.nn.softmax(block_logits)

    def get_joined_probabilities(self):
        return self.block_prob, self.direction_prob

    def get_direction_weights(self):
        return self.weights["w_dir"], self.biases["b_dir"]
