#!/usr/bin/python3

import sys
import tensorflow as tf
import os


@tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32)])
def while_loop_with_invariant(n):
    invariant_var_input = tf.constant(1, dtype=tf.int32)

    def condition(counter, total_product, invariant_var_arg):
        return counter <= n

    def body(counter, total_product, invariant_var_arg):
        return [
            counter + 1,
            total_product * counter,
            invariant_var_arg
        ]

    # Initial values
    counter_initial = tf.constant(1, dtype=tf.int32)
    total_product_initial = tf.constant(1, dtype=tf.int32)

    # Execute the loop
    _, final_prod, invariant_var_output = tf.while_loop(
        condition,
        body,
        [counter_initial, total_product_initial, invariant_var_input],
        shape_invariants=[
            tf.TensorShape([1]),
            tf.TensorShape([1]),
            tf.TensorShape([1])
        ]
    )

    result = final_prod * invariant_var_output
    return result


tf_net = while_loop_with_invariant.get_concrete_function(n=tf.TensorSpec(shape=[1], dtype=tf.int32)).graph
tf.io.write_graph(tf_net, os.path.join(sys.argv[1], 'loop_with_invariant'), 'loop_with_invariant.pb', as_text=False)
