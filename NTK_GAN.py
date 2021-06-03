import jax.numpy as np

from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap

import functools



import neural_tangents as nt
from neural_tangents import stax

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(font_scale=1.3)

print("NTK running")


input_range = [-2, 2]
num_data = 100
seed = 0
noise_scale = 0.1

key = random.PRNGKey(seed)


target_fn = lambda x: np.sin(x)


train_xs = random.uniform(key, (num_data, 1), minval=input_range[0], maxval=input_range[1])
train_ys = target_fn(train_xs)
train_ys += noise_scale * random.normal(key, (num_data, 1))
train_data = (train_xs, train_ys)

test_xs = np.linspace(input_range[0], input_range[1], num_data).reshape([-1, 1])
test_ys = target_fn(test_xs)
test_data = (test_xs, test_ys)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(8, W_std=1.5, b_std=0.05), stax.Relu(),
    stax.Dense(1, W_std=1.5, b_std=0.05)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn)


_, params = init_fn(key, (-1, 1))
print(f"params: {params}")
ys_pred = apply_fn(params, test_xs)
# print(f"ys_pred: {ys_pred}")


plt.plot(test_xs, test_ys)
plt.plot(test_xs, ys_pred)
plt.show()


