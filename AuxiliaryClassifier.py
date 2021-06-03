import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import time
import random
import pickle

import tensorflow as tf
import numpy as np
import scipy
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh


from collections import OrderedDict
from tqdm import tqdm

from utils.data import MOG_2D
from utils.misc import *
from utils.optim import RMSProp
from utils.logger import get_logger
from utils.deepviz import DeepVisuals_2D

try:
    import winsound  # for sound
except:
    print("Not using Windows")

from keras import backend as K
K.clear_session()

def Train_Auxiliary_Classifier(opt, plot=True, input_samples=None):

    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)


    title = f"Aux_{opt.data}_ds{opt.data_std}_sd{opt.seed}_{opt.layers}-{opt.neurons}_it{opt.iteration}_bs{opt.batch_size}_lr{opt.learning_rate}"

    root_dir = 'Aux_Models'
    os.makedirs(root_dir, exist_ok=True)

    model_save_folder = os.path.join(root_dir, title + time.strftime("_%Y%m%d_%H%M%S"))
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    """
    >>> Data
    """
    rng = np.random.RandomState(seed=opt.seed)
    data_generator = MOG_2D(opt.data, rng, std=opt.data_std, sample_per_mode=1000)
    n_modes = data_generator.n

    """
    >>> Model
    """
    def activation_fn(name):
        if name == 'elu':
            return tf.nn.elu
        elif name == 'relu':
            return tf.nn.relu
        elif name == 'tanh':
            return tf.nn.tanh

    def Classifier(x, n_hidden=opt.neurons, n_layer=opt.layers, reuse=False,
                      initializer=(tf.glorot_normal_initializer(seed=opt.seed) if opt.init=='xavier'
                      else tf.initializers.orthogonal(gain=1.0))):
        with tf.variable_scope("classifier", reuse=reuse):
            for i in range(n_layer):
                x = tf.layers.dense(x, n_hidden, activation=activation_fn(opt.act), kernel_initializer=initializer)
            x = tf.layers.dense(x, n_modes + 1, activation=None, kernel_initializer=initializer)
        return x


    """
    >>> Pipeline
    """
    all_samples_ = tf.placeholder(tf.float32, [None, opt.x_dim])
    all_labels_onehot_ = tf.placeholder(tf.int32, [None, data_generator.n + 1])

    classifier_logits_ = Classifier(all_samples_)
    loss_classification_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=all_labels_onehot_, logits=classifier_logits_))

    classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier")
    optimizer = tf.train.AdamOptimizer(opt.learning_rate)
    train_op = optimizer.minimize(loss_classification_, var_list=classifier_vars)


    """
    >>> Saver
    """
    saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=8, write_version=tf.train.SaverDef.V1)


    """
    >>> Graph
    """
    # tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    """
    >>> Training
    """
    feed_dict = {all_samples_: data_generator.all_samples, all_labels_onehot_: data_generator.all_labels_onehot}
    sess.run(tf.global_variables_initializer())

    def Plot_result(sess, i, plot=True):
        if input_samples is None:
            loss_classification, classifier_logits = sess.run([loss_classification_, classifier_logits_], feed_dict=feed_dict)
            pred_all_labels = np.argmax(classifier_logits, axis=1)
            print(f"iter: {i}, loss: {loss_classification}")
            if plot:
                plt.figure(figsize=(14, 14))
                plt.scatter(data_generator.all_samples[:, 0], data_generator.all_samples[:, 1], \
                            c=pred_all_labels, cmap=data_generator.cm)
                plt.axes().set_aspect('equal')
                plt.savefig(os.path.join(model_save_folder, f"iter_{i}_{opt.iteration}.png"), dpi=300)
        else:
            classifier_logits = sess.run([classifier_logits_], feed_dict={all_samples_: input_samples})
            pred_all_labels = np.argmax(classifier_logits, axis=1)



        return loss_classification, classifier_logits, pred_all_labels


    if opt.resume:
        print("Checkpoint", os.path.join(root_dir, opt.resume))
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(root_dir, opt.resume))
        print("latest_ckpt", latest_ckpt)
        saver.restore(sess, latest_ckpt)
        # saver.restore(sess, tf.train.load_checkpoint(os.path.join(model_save_folder, opt.resume)))
        loss_classification, classifier_logits, pred_all_labels = Plot_result(sess, -1, plot=plot)
    else:
        print("data_generator.all_samples.shape", data_generator.all_samples.shape)
        print("data_generator.all_labels_onehot.shape", data_generator.all_labels_onehot.shape)
        for i in range(opt.iteration + 1):
            if i % opt.plot_iter == 0:
                Plot_result(sess, i)

            if i % opt.save_iter == 0:
                saver.save(sess, os.path.join(model_save_folder, "Checkpoint"), global_step=i)

            sess.run(train_op, feed_dict=feed_dict)

        loss_classification, classifier_logits, pred_all_labels = Plot_result(sess, opt.iteration, plot=False)

    sess.close()
    return loss_classification, classifier_logits, pred_all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", "--iter", type=int, default=1000, help="number of iterations of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.0001, help="classifier learning rate")
    parser.add_argument("--neurons", type=int, default=32, help="dimension of hidden units")
    parser.add_argument("--layers", type=int, default=5, help="num of hidden layer")
    parser.add_argument("--act", type=str, default="relu", help="which activation function for disc")  # elu, relu, tanh
    parser.add_argument("--x_dim", type=int, default=2, help="data dimension")
    parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
    parser.add_argument("--data", type=str, default="grid", help="which dataset")
    parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--plot_iter", type=int, default=100, help="number of iter to plot")
    parser.add_argument("--save_iter", type=int, default=1000, help="number of iter to save")

    opt = parser.parse_args()
    print("\n")
    print("<" * 100)
    print(opt)
    print(">" * 100)
    print("\n")
    Train_Auxiliary_Classifier(opt)