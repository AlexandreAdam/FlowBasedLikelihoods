import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    # Example of a reverse-mode autodiff vector-Jacbian product (commonly known as backprop)
    x = tf.constant([[2.0, 3.0], [1.0, 4.0]])
    dense = tf.keras.layers.Dense(2)
    dense.build([None, 2])
    with tf.GradientTape(persistent=True) as tape:
        # forward phase -> record operation
        tape.watch(x)
        y = tf.reduce_sum(x**2, axis=1, keepdims=True)
    # backward phase -> compute gradient using graph of recorded operations on watched variables
    vjp = tape.gradient(y, x)
    print("vjp=",vjp.shape) # this is implicitely the vjp 
    print(vjp)
    jac = tape.jacobian(y, x)
    print("jac=", jac.shape)
    explicit_vjp = tf.reduce_sum(jac, axis=[0])
    print("explicit_vjp=", explicit_vjp.shape)
    print(explicit_vjp)
    delta = tf.reduce_max(tf.abs(vjp - explicit_vjp)).numpy()
    print(f"delta={delta}")

    # Now the same thing but explicitely with a random vector
    eps = tf.random.normal(shape=x.shape)
    random_vjp =  tape.gradient(y, x, eps)
    print("random vjp",random_vjp)


