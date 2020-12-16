"""
=============================================
Title: Continuous Normalizing Flow

Author(s): Alexandre Adam

Last modified: December 12, 2020

Description: Layer that solve the continuous transformation
    between Z (hidden random variable) to X (the observed random variable)
=============================================
"""
import tensorflow as tf
import tensorflow_probability as tfp

class CNF(tf.Module):

    def __init__(
            self, 
            odefunc,
            T=1., # integration final time
            stiff=True, # whether the ode is stiff or non-stiff
            rtol=1e-3, 
            atol=1e-6, 
            first_step_size=1e-3, 
            safety_factor=0.9,
            min_step_size_factor=0.1,
            max_step_size_factor=10.,
            max_num_step=None,
            train_T=False, 
            name="CNF"):
        """
        Continuous Normalizing Flow (CNF) class takes and Ordinary Differential Equation 
        function (odefunc), which is essentially the update equation of the ODE, and integrate 
        it to the desired time (T) (built-in call method).
        """
        super(CNF, self).__init__(name=name)
        self.odefunc = odefunc
        if stiff:
            self.odesolver = tfp.math.ode.BDF(
                    rtol=rtol.
                    atol=atol,
                    first_step_size=first_step_size,
                    safety_factor=safety_factor,
                    min_step_size_factor=min_step_size_factor,
                    max_step_size_factor=max_step_size_factor,
                    max_num_step=max_num_step
                    )
        else:
            self.odesolver = tfp.math.ode.DormandPrince(
                    rtol=rtol,
                    atol=atol,
                    first_step_size=first_step_size,
                    safety_factor=safety_factor,
                    min_step_size_factor=min_step_size_factor,
                    max_step_size_factor=max_step_size_factor,
                    max_num_step=max_num_step
                    )
        self.rtol=rtol
        self.atol=atol

    def __call__(self, z, logpz=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = tf.zeros_like(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = tf.constant([0, 1.])
        if reverse:
            integration_times = tf.reverse_sequence(integration_times, 
                    seq_lengths=integration_times.shape[0], seq_axis=0)

        state_t = self.odesolver.solve(
                ode_fn=self.odefunc,
                initial_time=integration_times[0],
                initial_state=(z, _logpz),
                solution_times=integration_times[1:],
                jacobian_fn=1. #TODO
                )


