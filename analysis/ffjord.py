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
            num_samples=1, # cannot change this number for now 
            samples_distribution="rademacher",
            name="ffjord",
            dtype=tf.float32
            ):
        super(FFJORD, self).__init__(forward_min_event_ndims=0, name=name, dtype=dtype)
        self._dtype=dtype
        self.state_derivative_fn = state_derivative_fn
        self.ode_solver_fn = ode_solver_fn

        self.state_size = state_size

        if samples_distribution == "rademacher":
            self._sample_dist = tfp.random.rademacher
        elif samples_distribution == "normal":
            self._epsilon = tf.random.normal

        def _inverse_state_derivative_fn(t, state):
            return -self.state_derivative_fn(1. - t, state)

        self.inverse_state_derivative_fn = _inverse_state_derivative_fn

    # @tf.function
    def _augmented_dynamics(self, reverse=False):
        # Equation 4 of Gratwohl et al.
        if reverse:
            state_derivative_fn = self.inverse_state_derivative_fn
        else:
            state_derivative_fn = self.state_derivative_fn

        def augmented_dynamics_fn(time, augmented_x):
            state, _ = augmented_x
            random_samples = self._sample_dist(shape=[self.num_samples, *self.state_size])

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(state)
                # state_time_derivative is f(z, t; theta) in Gratwohl et al.
                state_time_derivative = state_derivative_fn(time, state)

            def trace(random_sample):
                # Compute vector-Jacobian product
                vjp = tape.gradient(state_time_derivative, state, random_sample)
                return vjp

            results = tf.map_fn(trace, random_samples)
            trace_estimate = tf.reduce_mean(results, axis=0)

            return state_time_derivative, -trace_estimate

        return augmented_dynamics_fn


    def augmented_forward(self, x):
        # x should be sampled from the base distribution

        # initial condition of equation 4
        log_det_jacobian = tf.zeros(x.shape, dtype=self._dtype)

        augmented_x = (x, log_det_jacobian)
        results = self.ode_solver_fn(
                ode_fn=self._augmented_dynamics(),
                initial_time=0.,
                initial_state=augmented_x,
                solution_times=[1.],
                )
        final_state = tf.nest.map_structure(lambda x: x[-1], results.states)
        return final_state

    def augmented_inverse(self, y):
        # initial_condition from equation 4 of Gratwohl et al.
        log_det_jacobian = tf.zeros(y.shape, dtype=self._dtype)

        augmented_y = (y, log_det_jacobian)
        results = self.ode_solver_fn(
                ode_fn=self._augmented_dynamics(reverse=True),
                initial_state=augmented_y,
                initial_time=1.,
                solution_times=[0.]
                )
        final_state = tf.nest.map_structure(lambda x: x[-1], results.states)
        return final_state

    def _forward(self, x):
        # Transform the distribution Z to X
        y, _ = self.augmented_forward(x)
        return y

    def _inverse(self, y):
        x, _ = self.augmented_inverse(y)
        return x

    def _forward_log_det_jacobian(self, x):
        _, forward_log_det_jac = self.augmented_forward(x)
        return forward_log_det_jac

    def _inverse_log_det_jacobian(self, y):
        _, inverse_log_det_jac = self.augmented_inverse(y)
        return inverse_log_det_jac

