"""
=============================================
Title: FFJORD training

Author(s): Alexandre Adam

Last modified: December 20, 2020

Description: Self contained script to train FFJORD
=============================================
"""
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import sklearn.datasets as skd
from moon_ffjord_tf import make_grid, set_limits, plot_panel, plot_density , plot_points
import matplotlib.pyplot as plt
import os

# moon test
TEST = True
DATASET_SIZE = 1024 * 8
BATCH_SIZE = 512
SAMPLE_SIZE = DATASET_SIZE

# BATCH_SIZE = 30000
NUM_HIDDEN = 8
NUM_OUTPUT = 2 if TEST else 37
NUM_LAYERS = 6
NUM_EPOCHS = 10

ODE_INT = tfp.math.ode.DormandPrince( # adaptive step size 5th order Runge-Kutta method
        rtol=1e-6,
        atol=1e-5,
        first_step_size=0.05,
        # safety_factor=0.9,
        # min_step_size_factor=0.1,
        # max_step_size_factor=10.,
        # max_num_steps=100
        ) 

LR_SCHEDULE = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100], values=[1e-4, 1e-5]
        )
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR_SCHEDULE)


def mlp(num_hidden, num_layers, num_output):
    m = tf.keras.models.Sequential()
    for _ in range(num_layers):
        m.add(tf.keras.layers.Dense(num_hidden))
        m.add(tf.keras.layers.Activation("tanh")) # activation must me Lipschitz continuous
    m.add(tf.keras.layers.Dense(num_output))
    return m


def main():
    global DATASET_SIZE # quick hack
    if TEST:
        moons = skd.make_moons(n_samples=DATASET_SIZE, noise=.06)[0]
        X = tf.data.Dataset.from_tensor_slices(moons.astype(np.float32))
        X =X.prefetch(tf.data.experimental.AUTOTUNE)
        X = X.cache()
        X = X.shuffle(DATASET_SIZE)
        X = X.batch(BATCH_SIZE)
    else:
        # extract dataset
        data = pd.read_csv("../powerspec.csv")
        data.pop("Unnamed: 0")
        ell_bins = [f"ell{i}" for i in range(37)]
        ell = np.logspace(np.log10(500), np.log10(5000), 37)
        X = data[ell_bins].to_numpy().astype(np.float32)
        # quick hack
        DATASET_SIZE = len(data)

        # preprocessing (Cholesky decomposition)
        mean = np.mean(X, axis=0)
        centered_X = X - mean
        cov = np.cov(X.T)
        L = np.linalg.cholesky(np.linalg.inv(cov))
        X = centered_X @ L

        # make tensorflow dataset
        X = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
        X = X.prefetch(tf.data.experimental.AUTOTUNE)
        X = X.shuffle(len(data))
        X = X.batch(BATCH_SIZE)

    model = mlp(NUM_HIDDEN, NUM_LAYERS, NUM_OUTPUT)
    model.build([None, NUM_OUTPUT])

    # instantiate an isotropic MVN as a base distribution
    base_loc = np.zeros(NUM_OUTPUT).astype(np.float32)
    base_sigma = np.ones(NUM_OUTPUT).astype(np.float32)
    base_distribution = tfp.distributions.MultivariateNormalDiag(base_loc, base_sigma)

    def full_augmented_inverse(samples, f_theta, num_hutchinson=1):
        """
        Unbiased stochastic log-density estimation using the FFJORD model
        ### Reference:
            Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). 
            Ffjord: Free-Form Continuous Dynamics for Scalable Reversible Generative Models. ArXiv.
            see algorithm 1

        Alse, compute the loss gradient to the model parameters to be backpropagated through the model
        ### Reference: 
            Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). 
            Neural ordinary differential equations. In arXiv.
        """
        # draw samples
        epsilon = tf.random.normal(shape=[num_hutchinson] + samples.shape)
        theta = model.trainable_weights

        def augmented_dynamics(t, augmented_state):
            state, _, grad_theta = augmented_state
            # compute the state derivative with reverse-mode autodiff
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(state)
                tape.watch(theta)
                state_derivative = f_theta(state)
                instant_loss_derivative = tf.reduce_mean(state_derivative, axis=0)

            def estimate_trace(random_sample):
                # estimate vjp with a single random vector
                vjp = tape.gradient(state_derivative, state, random_sample)
                trace = tf.einsum("...i, ...i -> ...", vjp, random_sample)  # trace broadcasted over batch dimension
                return trace

            # aggregate the mean of the results (Hutchinson estimator)
            traces = tf.map_fn(estimate_trace, epsilon)
            trace = tf.reduce_mean(traces, axis=0)

            # Compute the instant loss gradient derivative relative to theta
            loss_grad = tape.gradient(instant_loss_derivative, theta)

            return [state_derivative, -trace, [-g for g in loss_grad]]

        loss_grad = [tf.zeros(_theta.shape) for _theta in theta]  # loss gradient relative to model parameters at time t_1
        adjoint = tf.zeros(samples.shape[0])  # delta of the log_density at t_1
        initial_conditions = [samples, adjoint, loss_grad]
        # integrate backward in time to get z(0) and log p(z(0))
        results = ODE_INT.solve(
                ode_fn=augmented_dynamics,
                initial_time=0.,
                solution_times=[1.],
                initial_state=initial_conditions
                )
        state, delta_log_density, loss_grad = tf.nest.map_structure(lambda x: x[-1], results.states)
        log_density_x = base_distribution.log_prob(state) - delta_log_density
        return log_density_x, loss_grad

    def augmented_forward(samples, f_theta, num_hutchinson=1):
        """
        Sample from base distribution and integrate forward the augmented dynamics (with exact jacobian trace)
        scales quadratically with the number of dimensions
        """
        epsilon = tf.random.normal(shape=[num_hutchinson] + list(samples.shape))

        def augmented_dynamics(t, augmented_state):
            state, _ = augmented_state
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(state)
                state_derivative = f_theta(state)

            def estimate_trace(random_sample):
                # estimate vjp with a single random vector
                vjp = tape.gradient(state_derivative, state, random_sample)
                trace = tf.einsum("...i, ...i -> ...", vjp, random_sample)  # trace broadcasted over batch dimension
                return trace

            # aggregate the mean of the results (Hutchinson estimator)
            traces = tf.map_fn(estimate_trace, epsilon)
            trace = tf.reduce_mean(traces, axis=0)

            return [state_derivative, -trace]

        s0 = [samples, tf.zeros(samples.shape[0])]
        results = ODE_INT.solve(
                ode_fn=augmented_dynamics,
                initial_time=0.,
                solution_times=[1.],
                initial_state=s0
                )
        state_t, delta_log_density =  tf.nest.map_structure(lambda x: x[-1], results.states)
        log_prob = base_distribution.log_prob(samples) - delta_log_density
        return state_t, log_prob
        
    def loss_function(log_density):
        return -tf.reduce_mean(log_density)

    # @tf.function
    def get_samples():
        base_distribution_samples = base_distribution.sample(SAMPLE_SIZE)
        transformed_samples, _ = augmented_forward(base_distribution_samples, model)
        return base_distribution_samples, transformed_samples

    grid = make_grid(-3, 3, -3, 3, 4, 100)

    # @tf.function
    def get_transformed_grid():
        transformed_grid, _ = augmented_forward(grid, model)
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
    plt.savefig(f"../results/moon_myffjord_example/panel_myffjord_{panel_id:02d}.png")

    # train the model with the full augmented inverse function
    step = 0 # step counter
    losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"\nepoch {epoch+1}/{NUM_EPOCHS}")
        pb_i = tf.keras.utils.Progbar(DATASET_SIZE, stateful_metrics=["loss"])
        base_samples, transformed_samples = get_samples()
        transformed_grid = get_transformed_grid()
        evaluation_samples.append(
            (base_samples, transformed_samples, transformed_grid))
        for samples in X:

            log_density, loss_grad = full_augmented_inverse(samples, model)
            loss = loss_function(log_density)
            losses.append(loss)
            # regularization/gradient clipping if necessary
            OPTIMIZER.apply_gradients(zip(loss_grad, model.trainable_weights))
            step += 1
            values = [('loss', loss)]
            pb_i.add(BATCH_SIZE, values=values)

    for panel_id in range(1, len(evaluation_samples)):
        panel_data = evaluation_samples[panel_id]
        fig, axarray = plt.subplots(
          1, 4, figsize=(16, 6))
        plot_panel(grid, panel_data[0], panel_data[2], panel_data[1], moons, axarray)
        plt.tight_layout()
        plt.savefig(f"../results/moon_myffjord_example/panel_myffjord_{panel_id:02d}.png")
        plt.close("all")

    # learning curve
    plt.figure()
    plt.plot(losses, "k-")
    plt.xlabel("epochs")
    plt.ylabel(r"$-\mathbb{E}_{p(\mathbf{x})} \log q_\theta (\mathbf{x})$")
    plt.title("Learning curve")
    plt.savefig("../results/moon_myffjord_example/lr_curve_myffjord.png")


if __name__ == "__main__":
    if not os.path.exists("../results/moon_myffjord_example"):
        os.mkdir("../results/moon_myffjord_example")
    main()
