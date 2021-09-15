import numpy as np
import tensorflow as tf
from data_load import make_environments
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json


def adv_gradient(f, g, x, y):
    with tf.GradientTape(persistent=True) as grad:
        grad.watch(x)
        logit_y = f(x)
        logit_z = g(x)

    p = tf.sigmoid(logit_y)
    df = grad.gradient(logit_y, x)
    dg = grad.gradient(logit_z, x)
    dg, _ =  tf.linalg.normalize(dg, axis = 1)
    dot_df_dg = tf.reduce_sum(df * dg, axis = 1)
    return tf.reshape(tf.reshape((p - y), (-1, )) * dot_df_dg, (-1, 1)) * dg




if __name__ == '__main__':
    envs, test = make_environments()




    x_train, y_train = envs[1]
    x_test, y_test = test

    i = int(float(sys.argv[1]))

    import itertools
    gradient_attacks_v = [10, 20, 30]
    gradient_attack_step_v = [10, 1, 0.1]
    steps_v = [5000, 10000, 20000]
    learning_rate_v = [1, 0.1, 0.01]

    grid = list(itertools.product(gradient_attacks_v, gradient_attack_step_v, steps_v, learning_rate_v))

    gradient_attacks, gradient_attack_step, steps, learning_rate = grid[i]
    batch_size = 250

    adv_network = tf.keras.Sequential(
        [tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)]
    )


    X_train = tf.cast(x_train, dtype = tf.float32)
    Y_train = tf.cast(y_train.reshape((-1, 1)), dtype = tf.float32)



    batch = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    batch = batch.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    batch = batch.take(steps)

    optimizer = tf.optimizers.Adagrad(learning_rate)

    cluster_predictor = tf.keras.models.load_model(filepath='cluster_predictor')


    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    ba = tf.keras.metrics.BinaryAccuracy(threshold=0)


    for step, (x, y) in enumerate(batch, 1):

        if step > 500:
            for i in range(gradient_attacks):
                x = x + gradient_attack_step * adv_gradient(adv_network, cluster_predictor, x, y)/((i+1) ** (2/3))

        with tf.GradientTape() as g_adv_net:
            loss_batch = bce(y, adv_network(x))

        trainable_variables = adv_network.trainable_variables
        gradients = g_adv_net.gradient(loss_batch, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        if step % 200 == 0:
            test_error = ba(y_test.reshape((-1, 1)), adv_network(x_test)).numpy()
            print(f'Test accuracy after {step} step is {test_error}')


    test_accuracy = ba(y_test.reshape((-1, 1)), adv_network(x_test)).numpy()
    return_dict = {'gradient-attcks': gradient_attacks, 'gradient-attack-steps': gradient_attack_step,
     'learning_step': learning_rate, 'steps': steps, 'test_accuracy': test_accuracy}

    with open(f'~/projects/coloured-mnist/test-errors/{i}.json', 'w') as f:
        json.dump(return_dict, f)
    

