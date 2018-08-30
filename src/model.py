from typing import Tuple

import numpy as np

import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.python.keras import backend as K


# values for LunarLanver-v2
ADIM   = 2
ADTYPE = np.float32
SDIM   = 8
SDTYPE = np.float32


class Actor(object):
    def __init__(self,
                 hidden: Tuple[int, int]=(400, 300),
                 a_dim: int=ADIM,
                 a_scale: np.ndarray=np.asarray([[-1, -1], [1, 1]], dtype=np.float32),
                 s_dim: int=SDIM,
                 lr: float=1e-4,
                 tau: float=0.001) -> None:
        
        self.hidden  = hidden
        self.a_dim   = a_dim
        self.a_scale = (a_scale[1, :] - a_scale[0, :]).astype(ADTYPE).reshape(1, -1)
        self.a_bias  = np.asarray(a_scale[0, :], dtype=ADTYPE).reshape(1, -1)
        self.s_dim   = s_dim
        
        # ----- build a main NN and copy its parameters to a target network -----
        s = tf.placeholder(dtype=tf.float32, shape=(None, self.s_dim), name='state')
        dq_da = tf.placeholder(dtype=tf.float32, shape=(None, self.a_dim), name='action')

        # main neural network
        a, main_scope = self.build_nn(state=s, training=True, name='actor_main')

        # training ops
        train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope)
        batch_size   = tf.cast(tf.shape(s)[0], tf.float32)
        gradients    = tf.gradients(a, train_params, -dq_da / batch_size)
        optimizer    = tf.train.AdamOptimizer(learning_rate=lr)
        extra_ops    = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=main_scope)
        with tf.control_dependencies(extra_ops):
            updates = optimizer.apply_gradients(zip(gradients, train_params))

        self.act   = K.function(inputs=[s], outputs=[a])
        self.train = K.function(inputs=[s, dq_da], outputs=[], updates=[updates])

        a_t, target_scope = self.build_nn(state=s, training=False, name='actor_target')
        
        # graph for target network update
        main_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=main_scope)
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
        init_updates  = tf.group(*[tf.assign(t, m) for m, t in zip(main_params, target_params)])
        soft_updates  = tf.group(*[tf.assign(t, tau * m + (1. - tau) * t) for m, t in zip(main_params, target_params)])
        
        self.target_act     = K.function(inputs=[s], outputs=[a_t])
        self._target_init   = K.function(inputs=[], outputs=[], updates=[init_updates])
        self._target_update = K.function(inputs=[], outputs=[], updates=[soft_updates])

    def target_init(self):
        self._target_init([])

    def target_update(self):
        self._target_update([])

    def build_nn(self, state, training, name):
        with tf.variable_scope(name):
            var_scope = tc.framework.get_name_scope()

            x = state
            fan_in = self.s_dim
            for h in self.hidden:
                w_bound = 1. / np.sqrt(fan_in)
                initializer = tf.initializers.random_uniform(minval=-w_bound, maxval=w_bound)
                x = tf.layers.dense(inputs=x, units=h, kernel_initializer=initializer)
                #x = tc.layers.batch_norm(x, center=True, scale=True, is_training=training)
                x = tf.nn.relu(x)

                fan_in = h

            # Weights of the final layer are initialized by Uniform[-3e-3, 3e-3]
            w_bound = 3e-3
            initializer = tf.initializers.random_uniform(minval=-w_bound, maxval=w_bound)
            x = tf.layers.dense(inputs=x, units=self.a_dim, kernel_initializer=initializer)
            x = tf.nn.tanh(x)
            a = tf.add(tf.multiply(tf.add(x, 1), 0.5 * self.a_scale), self.a_bias)

        return a, var_scope


class Critic(object):
    def __init__(self,
                 hidden: Tuple[int, int]=(400, 300),
                 a_dim: int=ADIM,
                 s_dim: int=SDIM,
                 lr: float=1e-3,
                 tau: float=0.001,
                 reg: float=1e-2) -> None:
        self.hidden = hidden
        self.a_dim  = a_dim
        self.s_dim  = s_dim

        # ----- build a main NN and copy its parameters to a target network -----
        a = tf.placeholder(dtype=tf.float32, shape=(None, self.a_dim), name='action')
        s = tf.placeholder(dtype=tf.float32, shape=(None, self.s_dim), name='state')
        y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Q_estimate')
        # main neural network
        q, main_scope = self.build_nn(state=s, action=a, training=True, name='critic_main')

        # training ops
        train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope)
        error_loss   = tf.reduce_mean(tf.pow(tf.subtract(y, q), 2), axis=0)
        weights      = [var for var in train_params if 'kernel' in var.name]
        l2_reg_loss  = tc.layers.apply_regularization(tc.layers.l2_regularizer(reg), weights_list=weights)
        total_loss   = tf.add(error_loss, l2_reg_loss)
        gradients    = tf.gradients(total_loss, train_params)
        optimizer    = tf.train.AdamOptimizer(learning_rate=lr)
        extra_ops    = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=main_scope)
        with tf.control_dependencies(extra_ops):
            updates = optimizer.apply_gradients(zip(gradients, train_params))

        self.dq_da = K.function(inputs=[s, a], outputs=[tf.gradients(q, a)[0]])
        self.q_val = K.function(inputs=[s, a], outputs=[q])
        self.train = K.function(inputs=[s, a, y], outputs=[], updates=[updates])

        q_t, target_scope = self.build_nn(state=s, action=a, training=False, name='critic_target')

        # graph for target network update
        main_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=main_scope)
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
        init_updates  = tf.group(*[tf.assign(t, m) for m, t in zip(main_params, target_params)])
        soft_updates  = tf.group(*[tf.assign(t, tau * m + (1. - tau) * t) for m, t in zip(main_params, target_params)])

        self.target_q_val   = K.function(inputs=[s, a], outputs=[q_t])
        self._target_init   = K.function(inputs=[], outputs=[], updates=[init_updates])
        self._target_update = K.function(inputs=[], outputs=[], updates=[soft_updates])

    def target_init(self):
        self._target_init([])

    def target_update(self):
        self._target_update([])

    def build_nn(self, state, action, training, name):
        with tf.variable_scope(name):
            var_scope = tc.framework.get_name_scope()

            x = state
            fan_in = self.s_dim

            w_bound = 1. / np.sqrt(fan_in)
            initializer = tf.initializers.random_uniform(minval=-w_bound, maxval=w_bound)
            x = tf.layers.dense(inputs=state, units=self.hidden[0], kernel_initializer=initializer)
            #x = tc.layers.batch_norm(x, center=True, scale=True, is_training=training)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=1)
            fan_in = self.hidden[0] + self.a_dim
            for h in self.hidden[1:]:
                w_bound = 1. / np.sqrt(fan_in)
                initializer = tf.initializers.random_uniform(minval=-w_bound, maxval=w_bound)
                x = tf.layers.dense(inputs=x, units=h, kernel_initializer=initializer)
                #x = tc.layers.batch_norm(x, center=True, scale=True, is_training=training)
                x = tf.nn.relu(x)

                fan_in = h

            # Weights of the final layer are initialized by Uniform[-3e-3, 3e-3]
            w_bound = 3e-3
            initializer = tf.initializers.random_uniform(minval=-w_bound, maxval=w_bound)
            q = tf.layers.dense(inputs=x, units=1, kernel_initializer=initializer)

        return q, var_scope
