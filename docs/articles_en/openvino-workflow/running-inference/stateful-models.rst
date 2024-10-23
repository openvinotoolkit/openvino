Stateful models and State API
==============================

.. toctree::
   :maxdepth: 1
   :hidden:

   stateful-models/obtaining-stateful-openvino-model

A "stateful model" is a model that implicitly preserves data between two consecutive inference
calls. The tensors saved from one run are kept in an internal memory buffer called a
"state" or a "variable" and may be passed to the next run, while never being exposed as model
output. In contrast, for a "stateless" model to pass data between runs, all produced data is
returned as output and needs to be handled by the application itself for reuse at the next
execution.

.. image:: ../../assets/images/stateful_model_example.svg
   :alt: example comparison between stateless and stateful model implementations
   :align: center
   :scale: 90 %

What is more, when a model includes TensorIterator or Loop operations, turning it to stateful
makes it possible to retrieve intermediate values from each execution iteration (thanks to the
LowLatency transformation). Otherwise, the whole set of their executions needs to finish
before the data becomes available.

Text generation is a good usage example of stateful models, as it requires multiple inference
calls to output a complete sentence, each run producing a single output token. Information
from one run is passed to the next inference as a context, which may be handled by a stateful
model natively. Potential benefits for this, as well as other scenarios, may be:

1. **model execution speedup** - data in states is stored in the optimized form for OpenVINO
   plugins, which helps to execute the model more efficiently. Importantly, *requesting data
   from the state too often may reduce the expected performance gains* or even lead to
   losses. Use the state mechanism only if the state data is not accessed very frequently.

2. **user code simplification** - states can replace code-based solutions for such scenarios
   as giving initializing values for the first inference call or copying data from model
   outputs to inputs. With states, OpenVINO will manage these cases internally, additionally
   removing the potential for additional overhead due to data representation conversion.

3. **data processing** - some use cases require processing of data sequences.
   When such a sequence is of known length and short enough, you can process it with RNN-like
   models that contain a cycle inside. When the length is not known, as in the case of online
   speech recognition or time series forecasting, you can divide the data in small portions and
   process it step-by-step, which requires addressing the dependency between data portions.
   States fulfil this purpose well: models save some data between inference runs, when one
   dependent sequence is over, the state may be reset to the initial value and a new sequence
   can be started.


OpenVINO Stateful Model Representation
######################################

To make a model stateful, OpenVINO replaces looped pairs of `Parameter` and `Result` with its
own two operations:

* ``ReadValue`` (:doc:`see specs <../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/read-value-6>`)
  reads the data from the state and returns it as output.
* ``Assign`` (:doc:`see specs <../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/assign-6>`)
  accepts the data as input and saves it in the state for the next inference call.

Each pair of these operations works with **state**, which is automatically saved between
inference runs and can be reset when needed. This way, the burden of copying data is shifted
from the application code to OpenVINO and all related internal work is hidden from the user.

There are three methods of turning an OpenVINO model into a stateful one:

* :doc:`Optimum-Intel <../../learn-openvino/llm_inference_guide/llm-inference-hf>` - the most user-friendly option. All necessary optimizations
  are recognized and applied automatically. The drawback is, the tool does not work with all
  models.

* :ref:`MakeStateful transformation <ov_ug_make_stateful>` - enables the user to choose which
  pairs of Parameter and Result to replace, as long as the paired operations are of the same
  shape and element type.

* :ref:`LowLatency2 transformation <ov_ug_low_latency>` - automatically detects and replaces
  Parameter and Result pairs connected to hidden and cell state inputs of LSTM/RNN/GRU operations
  or Loop/TensorIterator operations.



.. _ov_ug_stateful_model_inference:

Running Inference of Stateful Models
#####################################

For the most basic applications, stateful models work out of the box. For additional control,
OpenVINO offers a dedicated API, whose methods enable you to both retrieve and change data
saved in states between inference runs. OpenVINO runtime uses ``ov::InferRequest::query_state``
to get the list of states from a model and the ``ov::VariableState`` class to operate with
states.

| **`ov::InferRequest` methods:**
|   ``std::vector<VariableState> query_state();`` - gets all available states for the given
    inference request
|   ``void reset_state()`` - resets all States to their default values
|
| **`ov::VariableState` methods:**
|   ``std::string get_name() const`` - returns name(variable_id) of the corresponding
    State(Variable)
|   ``void reset()`` - resets the state to the default value
|   ``void set_state(const Tensor& state)`` - sets a new value for the state
|   ``Tensor get_state() const`` - returns the current value of the state


| **Using multiple threads**
| Note that if multiple independent sequences are involved, several threads may be used to
  process each section in its own infer request. However, using several infer requests
  for one sequence is not recommended, as the state would not be passed automatically. Instead,
  each run performed in a different infer request than the previous one would require the state
  to be set "manually", using the ``ov::VariableState::set_state`` method.

.. image:: ../../assets/images/stateful_model_init_subgraph.svg
   :alt: diagram of how initial state value is set or reset
   :align: center
   :scale: 100 %

| **Resetting states**
| Whenever it is necessary to set the initial value of a state or reset it, an initializing
| subgraph for the ReadValue operation and a special ``reset`` method are provided.
| A case worth mentioning here is, if you decide to reset, query for states, and then retrieve
| state data. It will result in undefined values and so, needs to be avoided.


Stateful Model Application Example
###################################

Here is a code example demonstrating inference of three independent sequences of data.
One infer request and one thread are used. The state should be reset between consecutive
sequences.

.. tab:: C++

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
         :language: cpp
         :fragment: [ov:state_api_usage]


You can find more examples demonstrating how to work with states in other articles:

* `LLM Chatbot notebook <../../notebooks/stable-zephyr-3b-chatbot-with-output.html>`__
* :doc:`Serving Stateful Models with OpenVINO Model Server <../../openvino-workflow/model-server/ovms_docs_stateful_models>`
