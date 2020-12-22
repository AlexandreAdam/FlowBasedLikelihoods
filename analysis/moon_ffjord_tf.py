"""
=============================================
Title: FFJORD Moons Example

Author(s): Alexandre Adam

Last modified: December 12, 2020

Description: Example of FFJORD with tensorflow probability and 
    moons dataset. It follows (line by line) the example from 
    https://www.tensorflow.org/probability/examples/FFJORD_Demo.

    It also serves as a baseline for our implementation of FFJORD.
=============================================
"""

import tensorflow as tf
import tensorflow_probability as tfp
import sklearn.datasets as skd
import matplotlib.pyplot as plt
import sonnet as snt # abstraction module from DeepMind to build NeuralNetworks
import numpy as np
import seaborn as sns
import tqdm
from scipy.stats import kde
import os
from ffjord import FFJORD


tfb = tfp.bijectors
tfd = tfp.distributions

# dataset 
DATASET_SIZE = 1024 * 8
BATCH_SIZE = 256
SAMPLE_SIZE = DATASET_SIZE

# FFJORD hyperparameters
LR = 1e-2
NUM_EPOCHS = 80
STACKED_FFJORDS = 4
NUM_HIDDEN = 8
NUM_LAYERS = 3
NUM_OUTPUT = 2


class MLP_ODE(snt.Module):
    """Multi-layer NN ode_fn."""
    def __init__(self, num_hidden, num_layers, num_output, name='mlp_ode'):
        super(MLP_ODE, self).__init__(name=name)
        self._num_hidden = num_hidden
        self._num_output = num_output
        self._num_layers = num_layers
        self._modules = []
        for _ in range(self._num_layers - 1):
            self._modules.append(snt.Linear(self._num_hidden))
            self._modules.append(tf.math.tanh)
            self._modules.append(snt.Linear(self._num_output))
            self._model = snt.Sequential(self._modules)

    def __call__(self, t, inputs):
        inputs = tf.concat([tf.broadcast_to(t, inputs.shape), inputs], -1)
        return self._model(inputs)

def make_grid(xmin, xmax, ymin, ymax, gridlines, pts):
      xpts = np.linspace(xmin, xmax, pts)
      ypts = np.linspace(ymin, ymax, pts)
      xgrid = np.linspace(xmin, xmax, gridlines)
      ygrid = np.linspace(ymin, ymax, gridlines)
      xlines = np.stack([a.ravel() for a in np.meshgrid(xpts, ygrid)])
      ylines = np.stack([a.ravel() for a in np.meshgrid(xgrid, ypts)])
      return np.concatenate([xlines.astype(np.float32), ylines.astype(np.float32)], 1).T

def plot_density(data, axis):
    x, y = np.squeeze(np.split(data, 2, axis=1))
    levels = np.linspace(0.0, 0.75, 10)
    kwargs = {'levels': levels}
    return sns.kdeplot(x, y=y, cmap="viridis", shade=True, 
                     thresh=0.05, ax=axis, **kwargs)


def plot_points(data, axis, s=10, color='b', label=''):
    x, y = np.squeeze(np.split(data, 2, axis=1))
    axis.scatter(x, y, c=color, s=s, label=label)


def plot_panel(
        grid, 
        samples, 
        transformed_grid, 
        transformed_samples,
        dataset, 
        axarray, 
        limits=True):
    if len(axarray) != 4:
        raise ValueError('Expected 4 axes for the panel')
    ax1, ax2, ax3, ax4 = axarray
    plot_points(data=grid, axis=ax1, s=20, color='black', label='grid')
    plot_points(samples, ax1, s=30, color='blue', label='samples')
    plot_points(transformed_grid, ax2, s=20, color='black', label='ode(grid)')
    plot_points(transformed_samples, ax2, s=30, color='blue', label='ode(samples)')
    ax3 = plot_density(transformed_samples, ax3)
    ax4 = plot_density(dataset, ax4)

    ax3.set_title("FFJORD density")
    ax4.set_title("Data density")
    ax1.set_title("Base distribution map")
    ax2.set_title("Transformed map")
    if limits:
        set_limits([ax1], -3.0, 3.0, -3.0, 3.0)
        set_limits([ax2], -2.0, 3.0, -2.0, 3.0)
        set_limits([ax3, ax4], -1.5, 2.5, -0.75, 1.25)


def set_limits(axes, min_x, max_x, min_y, max_y):
  if isinstance(axes, list):
    for axis in axes:
      set_limits(axis, min_x, max_x, min_y, max_y)
  else:
    axes.set_xlim(min_x, max_x)
    axes.set_ylim(min_y, max_y)


def main():
    # make dataset
    moons = skd.make_moons(n_samples=DATASET_SIZE, noise=.06)[0]
    moons_ds = tf.data.Dataset.from_tensor_slices(moons.astype(np.float32))
    moons_ds = moons_ds.prefetch(tf.data.experimental.AUTOTUNE)
    moons_ds = moons_ds.cache()
    moons_ds = moons_ds.shuffle(DATASET_SIZE)
    moons_ds = moons_ds.batch(BATCH_SIZE)

    # dataset visualization (in results directory)
    plt.figure(figsize=[8, 8])
    plt.scatter(moons[:, 0], moons[:, 1])
    plt.title("Moon dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("../results/moon_distribution.png")

    # instantiate a base distribution (2d)
    base_loc = np.array([0.0, 0.0]).astype(np.float32)
    base_sigma = np.array([0.8, 0.8]).astype(np.float32)
    base_distribution = tfd.MultivariateNormalDiag(base_loc, base_sigma)

    # Instantiate solver and trace augmentation function
    solver = tfp.math.ode.DormandPrince(atol=1e-5) # non-stiff solver
    ode_solve_fn = solver.solve
    trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson

    # Define bijector function (here, we stack 4 FFJORD models)
    bijectors = []
    for _ in range(STACKED_FFJORDS):
        mlp_model = MLP_ODE(NUM_HIDDEN, NUM_LAYERS, NUM_OUTPUT)
#       next_ffjord = tfb.FFJORD(
          # state_time_derivative_fn=mlp_model,
          # ode_solve_fn=ode_solve_fn,
          # trace_augmentation_fn=trace_augmentation_fn
          # )
        next_ffjord = FFJORD(
                state_derivative_fn=mlp_model,
                ode_solver_fn=ode_solve_fn,
                state_size=2
                )
        bijectors.append(next_ffjord)

    stacked_ffjord = tfb.Chain(bijectors[::-1])

    # transformed distribution from the bijector (stacked_ffjord)
    transformed_distribution = tfd.TransformedDistribution(
        distribution=base_distribution, 
        bijector=stacked_ffjord
        )

    # Utility function for training
    @tf.function
    def train_step(optimizer, target_sample):
        with tf.GradientTape() as tape:
            # expectation of the log probability of X given our model.
            loss = -tf.reduce_mean(transformed_distribution.log_prob(target_sample))
        variables = tape.watched_variables()
        gradients = tape.gradient(loss, variables)
        optimizer.apply(gradients, variables)
        return loss

    @tf.function
    def get_samples():
        base_distribution_samples = base_distribution.sample(SAMPLE_SIZE)
        transformed_samples = transformed_distribution.sample(SAMPLE_SIZE)
        return base_distribution_samples, transformed_samples

    grid = make_grid(-3, 3, -3, 3, 4, 100)

    @tf.function
    def get_transformed_grid():
        transformed_grid = stacked_ffjord.forward(grid)
        return transformed_grid

    # Plotting of progress
    evaluation_samples = []
    print("Getting samples")
    base_samples, transformed_samples = get_samples()
    transformed_grid = get_transformed_grid()
    evaluation_samples.append((base_samples, transformed_samples, transformed_grid))

    # Initial data transformation
    panel_id = 0
    panel_data = evaluation_samples[panel_id]
    fig, axarray = plt.subplots(
      1, 4, figsize=(16, 6))
    plot_panel(
        grid, panel_data[0], panel_data[2], panel_data[1], moons, axarray, False)
    plt.tight_layout()
    plt.savefig(f"../results/moon_ffjord_example/panel_myffjord_{panel_id:02d}.png")

    # Training
    learning_rate = tf.Variable(LR, trainable=False)
    optimizer = snt.optimizers.Adam(learning_rate)

    losses = []
    print("started training")
    for epoch in tqdm.trange(NUM_EPOCHS // 2):
        base_samples, transformed_samples = get_samples()
        transformed_grid = get_transformed_grid()
        evaluation_samples.append(
          (base_samples, transformed_samples, transformed_grid))
        for batch in moons_ds:
            losses.append(train_step(optimizer, batch))

    for panel_id in range(1, len(evaluation_samples)):
        panel_data = evaluation_samples[panel_id]
        fig, axarray = plt.subplots(
          1, 4, figsize=(16, 6))
        plot_panel(grid, panel_data[0], panel_data[2], panel_data[1], moons, axarray)
        plt.tight_layout()
        plt.savefig(f"../results/moon_ffjord_example/panel_myffjord_{panel_id:02d}.png")
        plt.close("all")

    # learning curve
    plt.figure()
    plt.plot(np.arange(NUM_EPOCHS//2), losses, "k-")
    plt.xlabel("epochs")
    plt.ylabel(r"-\mathbb{E}_{p(\mathbf{x})} \log q_\theta (\mathbf{x})")
    plt.title("Learning curve")
    plt.savefig("../results.moon_ffjord_example/lr_curve_myffjord.png")


if __name__ == "__main__":
    if not os.path.exists("../results/moon_ffjord_example"):
        os.mkdir("../results/moon_ffjord_example")
    main()

