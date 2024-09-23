Converting a JAX/Flax Model (Experimental)
==========================================


.. meta::
   :description: Learn how to convert a model from the
                 JAX/Flax format to the OpenVINO Model.


The ``openvino.convert_model`` function supports the following JAX/Flax model object types:

* ``jax._src.core.ClosedJaxpr``
* ``flax.linen.Module``

The ``jax._src.core.ClosedJaxpr`` object is created by tracing a Python function using the ``jax.make_jaxpr`` function.
Here is an example of ``jax._src.core.ClosedJaxpr`` object creation and conversion to an OpenVINO model:

.. code-block:: py
   :force:

   import jax
   import jax.numpy as jnp
   import openvino as ov

   # let us have some JAX function
   def jax_func(x, y):
       return jax.lax.tanh(jax.lax.max(x, y))

   # 1. Create ClosedJaxpr object
   x = jnp.array([1.0, 2.0])
   y = jnp.array([-1.0, 10.0])
   jaxpr = jax.make_jaxpr(jax_func)(x, y)
   # 2. Convert to OpenVINO
   ov_model = ov.convert_model(jaxpr)


Here is an example of the simplest ``flax.linen.Module`` model conversion:

.. code-block:: py
   :force:

   import flax.linen as nn
   import jax
   import jax.numpy as jnp
   import openvino as ov

   # let user have some Flax module
   class SimpleModule(nn.Module):
       features: int

       @nn.compact
       def __call__(self, x):
           return nn.Dense(features=self.features)(x)

   module = SimpleModule(features=4)

   # create example_input used for training
   example_input = jnp.ones((2, 3))

   # prepare parameters to initialize the module
   # they can be also loaded using pickle, flax.serialization
   key = jax.random.PRNGKey(0)
   params = module.init(key, example_input)
   module = module.bind(params)

   ov_model = ov.convert_model(module, example_input=example_input)


When using ``flax.linen.Module`` as an input model, ``openvino.convert_model`` requires the
``example_input`` parameter to be specified. Internally, it triggers model tracing during
the model conversion process, using the capabilities of the ``jax.make_jaxpr`` function.

The ``__call__`` method of ``flax.linen.Module`` object can also have extra custom flags
, like ``training``, in the input signature. In this case, it is required to create a helper function
that has an input signature without any extra custom flags or parameters, not related to input data.
Here is an example of handling such a case:

.. code-block:: py
   :force:

   import jax
   import jax.numpy as jnp
   import openvino as ov
   from flax import linen as nn
   from flax.core import freeze, unfreeze

   class SimpleModuleWithExtraFlag(nn.Module):
       features: int

       @nn.compact
       def __call__(self, x, training):
           x = nn.Dense(self.features)(x)
           x = nn.BatchNorm(use_running_average=not training)(x)
           return x


   # 1. Initialize the model
   module = SimpleModuleWithExtraFlag(features=10)
   key = jax.random.PRNGKey(0)
   input_data = jnp.ones((4, 5))  # Batch of 4 samples, each with 5 features
   params = module.init(key, input_data, training=False)

   # 2. Create helper function with only input data parameter
   def helper_function(x):
       return module.apply(params, x, training=False)

   # 3. Trace the helper function
   jaxpr = jax.make_jaxpr(helper_function)(input_data)

   # 4. Convert to OpenVINO
   ov_model = ov.convert_model(jaxpr)


.. note::

   The resulting OpenVINO IR model can be saved to drive with no additional, JAX-specific steps. 
   Use the standard ``ov.save_model(ov_model,'model.xml')`` command. 

Exporting a JAX/Flax Model to TensorFlow SavedModel Format
##########################################################

An alternative method of converting JAX/Flax models is exporting them to the TensorFlow SavedModel format
first, with ``jax.experimental.jax2tf.convert``,  and then converting the resulting SavedModel directory to OpenVINO IR,
with ``openvino.convert_model``. It can be considered a backup solution, if a model cannot be
converted directly, as described previously.

1. Refer to the `JAX and TensorFlow interoperation <https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md>`__
   guide to learn how to export models from JAX/Flax to TensorFlow SavedModel format.
2. Follow :doc:`Convert a TensorFlow model <convert-model-tensorflow>` chapter to produce an OpenVINO IR model.

Here is an illustration of using these two steps together:

.. code-block:: py
   :force:

   import flax.linen as nn
   import jax
   import jax.experimental.jax2tf as jax2tf
   import jax.numpy as jnp
   import openvino as ov
   import openvino as ov
   import tensorflow as tf

   # let user have some Flax module
   class SimpleModule(nn.Module):
       features: int

       @nn.compact
       def __call__(self, x):
           return nn.Dense(features=self.features)(x)

   flax_module = SimpleModule(features=4)

   # prepare parameters to initialize the module
   # they can be also loaded using pickle, flax.serialization
   example_input = jnp.ones((2, 3))
   key = jax.random.PRNGKey(0)
   params = flax_module.init(key, example_input)
   module = flax_module.bind(params)

   # 1. Export to SavedModel
   # create TF function and wrap it into TF Module
   tf_function = tf.function(jax2tf.convert(flax_module, native_serialization=False), autograph=False,
                             input_signature=[tf.TensorSpec(shape=[2, 3], dtype=tf.float32)])
   tf_module = tf.Module()
   tf_module.f = tf_function
   tf.saved_model.save(tf_module, './saved_model')

   # 2. Convert to OpenVINO
   ov_model = ov.convert_model('./saved_model')

.. note::

   As of version 0.4.15, it is required to pass the ``native_serialization=False`` parameter
   into ``jax2tf.convert`` for graph serialization mode. Without it, the created TensorFlow
   function will contain the embedded StableHLO modules that are not handled by the OpenVINO TensorFlow Frontend.
