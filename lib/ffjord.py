"""
=============================================
Title: Free Form Jacobian of Reversible Dynamics (FFJORD)

Author(s): Alexandre Adam

Last modified: December 13, 2020

Description: FFJORD built as a bijector (tensorflow probability)
=============================================
"""
import tensorflow as tf
import tensorflow_probability as tfp


class FFJORD(tfp.bijectors.Bijector):
    """
    We assume this bijector is a diffeomorphism between the random variable Z (prior) 
    to the observed data random variable (X).

    state_derviative_fn: should be the neurla net model or any transformation function
    ode_solver_fn: should be the solve method of the ode solver.

    Augmented dynamics are function that will return the state_time_deivative and the 
    trace estimate (see equation 4 of Gratwohl et al.). 


      #### References
      [1]:  Grathwohl, W., Chen, R. T., Betterncourt, J., Sutskever, I.,
            & Duvenaud, D. (2018). Ffjord: Free-form continuous dynamics for
            scalable reversible generative models. arXiv preprint arXiv:1810.01367.
            http://arxiv.org.abs/1810.01367
      [2]:  Hutchinson, M. F. (1989). A stochastic estimator of the trace of the
            influence matrix for Laplacian smoothing splines. Communications in
            Statistics-Simulation and Computation, 18(3), 1059-1076.
    """
    def __init__(
            self,
            state_derivative_fn, 
            ode_solver_fn, 
            state_size,
            num_samples=1, # for Hutchinson estimator
            samples_distribution="rademacher",
            name="ffjord"
            ):
        super(FFJORD, self).__init__(name=name)
        self.state_derivative_fn = state_derivative_fn
        self.ode_solver_fn = ode_solver_fn

        self.time_sequence = tf.constant([0, 1]) 

        # sample once the random vector for the Hutchinson Trace estimator
        if samples_distribution == "rademacher":
            self._epsilon = tfp.random.rademacher(shape=(num_samples, state_size))
        elif samples_distribution == "normal":
            self._epsilon = tf.random.normal(shape=(num_samples, state_size))

    # @tf.function
    def _augmented_dynamics(self, time, augmented_x, reverse=False):
        # Equation 4 of Gratwohl et al.
        if reverse:
            time_sequence = tf.reverse_sequence(self.time_sequence, seq_length=time_sequence.shape[0],
                    seq_axis=0)
        else:
            time_sequence = self.time_sequence

        state, _ = augmented_x
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(state)
            # state_time_derivative is f(z, t; theta) in Gratwohl et al.
            state_time_derivative = self.ode_solver_fn(
                    ode_fn=self.state_derivative_fn,
                    initial_time=time_sequence[0],
                    solution_times=time_sequence[1:]
                    )

        def estimate_trace(random_sample):
        # Compute Jacobian Vector product to save memory and computation cost
            jvp = tape.gradient(
                    target=state_time_derivative, 
                    source=state,
                    output_gradients=random_sample
                    )
            return random_sample * jvp

        trace_estimates = tf.map_fn(estimate_trace, self._epsilon)
        trace_estimates = tf.reduce_mean(trace_estimates, axis=0)

        return state_time_derivative, trace_estimates


    def augmented_forward(self, x):
        # x should be sampled from the base distribution

        # initial condition of equation 4
        log_det_jacobian = tf.zeros(x.shape, dtype=tf.keras.backend.floatx())

        augmented_x = (x, log_det_jacobian)
        results = self.ode_solver_fn(
                ode_fn=self._augmented_dynamics,
                initial_time=0.,
                initial_state=augmented_x,
                solution_times=[1.],
                )
        final_state = tf.nest.map_structure(lambda x: x[-1], results.state)
        return final_state

    def augmented_inverse(self, y):
        # initial_condition from equation 4 of Gratwohl et al.
        log_det_jacobian = tf.zeros(y.shape, dtype=tf.keras.backend.floatx())

        augmented_y = (y, log_det_jacobian)
        results = self.ode_solver_fn(
                ode_fn=self._augmented_dynamics,
                initial_time=1.,
                solution_times=[0.]
                )
        final_state = tf.nest.map_structure(lambda x: x[-1], results.state)
        return final_state

    def _forward(self, x):
        # Transform the distribution Z to X
        y, _ = self.augmented_forward(x)
        return y

    def _inverse(self, y):
        x, _ = self.augmented_inverse(y)
        return x








