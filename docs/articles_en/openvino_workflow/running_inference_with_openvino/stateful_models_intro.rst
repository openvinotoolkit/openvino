.. {#openvino_docs_OV_UG_stateful_models_intro}

Stateful models and State API
==============================

.. toctree::
    :maxdepth: 1
    :hidden:

    openvino_docs_OV_UG_ways_to_get_stateful_model

What is Stateful Model?
#######################

Stateful model is a model which implicitly keeps data from one inference call to the next inference call. Data is kept in internal runtime memory space usually called State or Variable.
In contrast to usual **stateless** model, which return all produced data as model outputs, **stateful**
model preserve part of the tensors saved in States without exposing them as model outputs.

The purpose of stateful models is to natively address a sequence processing tasks, like in text generation when one model inference produce a single output token,
and it is required to perform multiple inference calls to generate a complete output sentence.
Hidden state data from previous inference should be passed to the next inference as a context.
Usually the contextual data is not required to be accessed in the user application and should be just passed through to the next inference call manually using model API.
Stateful models simplifies programming of this scenario and unlocks additional performance potential of OpenVINO runtime.

.. _ov_ug_stateful_model_benefits:

OpenVINO Stateful Model Benefits
#################################

1. Speed up execution of Model
    Data in State is stored in the optimized form for OpenVINO plugins, which helps to execute model effectively.
    **Note:** Often requesting data from State might reduce the expected performance gains and even lead to losses,
    so, it is expected that State mechanism will be used when data stored in State is not accessed frequently.

2. Simplify user code
    Typical scenarios as providing initializing values for the first inference call or copying data from model's outputs to inputs in user code
    can be replaced with State. OpenVINO will manage these cases internally.

3. Specific scenarios
    Several use cases require processing of data sequences. When length of a sequence is known and small enough,
    we can process it with RNN like models that contain a cycle inside. But in some cases, like online speech recognition or time series
    forecasting, length of data sequence is unknown. Then data can be divided in small portions and processed step-by-step. But dependency
    between data portions should be addressed. For that, models save some data between inferences - state. When one dependent sequence is over,
    state should be reset to initial value and new sequence can be started.

OpenVINO Stateful Model Representation
######################################

OpenVINO contains ReadValue/Assign operations to make a model Stateful.
Each pair of ReadValue/Assign operates with State, known also as Variable,
which is an internal memory buffer to store tensor data during and between model inference calls.
ReadValue reads data tensor from State and returns it as output, Assign accepts data tensor as input and writes data to State
to save data for the next inference call.

OpenVINO has a special API to simplify work with Stateful models. State is automatically saved between inferences,
and there is a way to reset state when needed. You can also read state or set it to some new value between inferences.

.. image:: _static/images/stateful_model_example.svg
   :align: center

The left side of the picture shows the usual inputs and outputs to the model: Parameter/Result operations.
There is no direct connection from Result to Parameter and in order to copy data from output to input users need to put extra effort writing and maintaining additional code.
In addition, this may impose additional overhead due to data representation conversion.

Having operations such as ReadValue and Assign allows users to replace the looped Parameter/Result pairs of operations and shift the work of copying data to OpenVINO. After the replacement, the OpenVINO model no longer contains inputs and outputs with such names, all internal work on data copying is hidden from the user, but data from the intermediate inference can always be retrieved using State API methods.


.. image:: _static/images/stateful_model_init_subgraph.svg
   :align: center

In some cases, users need to set an initial value for State, or it may be necessary to reset the value of State at a certain inference to the initial value. For such situations, an initializing subgraph for the ReadValue operation and a special "reset" method are provided.

You can find more details on these operations in :doc:`ReadValue <openvino_docs_ops_infrastructure_ReadValue_6>` and
:doc:`Assign <openvino_docs_ops_infrastructure_Assign_6>` specification.

How to get OpenVINO Model with States
#########################################

* :ref:`Optimum-Intel<gen_ai_guide>`
   This is the most user-friendly way to get `the Benefits<_ov_ug_stateful_model_benefits>`
   from using Stateful models in OpenVINO.
   All necessary optimizations will be applied automatically inside Optimum-Intel tool.

* :ref:`Apply MakeStateful transformation.<ov_ug_make_stateful>`
   If after conversion from original model to OpenVINO representation, the resulting model contains Parameter and Result operations,
   which pairwise have the same shape and element type, the MakeStateful transformation can be applied to get model with states.

* :ref:`Apply LowLatency2 transformation.<ov_ug_low_latency>`
   If a model contains a loop that runs over some sequence of input data,
   the LowLatency2 transformation can be applied to get model with states.

.. _ov_ug_stateful_model_inference:

Stateful Model Inference
########################

The example below demonstrates inference of three independent sequences of data. State should be reset between these sequences.

One infer request and one thread will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step is done in one infer request and the second in another, state should be set in new infer request manually (using `ov::VariableState::set_state` method).

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_model_state_intro.cpp
         :language: cpp
         :fragment: [ov:state_api_usage]

You can find more powerful examples demonstrating how to work with models with states in speech sample and demo. 
Descriptions can be found in :doc:`Samples Overview<openvino_docs_OV_UG_Samples_Overview>`

.. _ov_ug_state_api:

OpenVINO State API
##################

OpenVINO runtime has the `ov::InferRequest::query_state` method  to get the list of states from a model and `ov::VariableState` class to operate with states.
Below you can find brief description of methods and the example of how to use this interface.

**`ov::InferRequest` methods:**

* `std::vector<VariableState> query_state();`
    allows to get all available stats for the given inference request.

* `void reset_state()`
    allows to reset all States to their default values.

**`ov::VariableState` methods:**

* `std::string get_name() const`
    returns name(variable_id) of the according State(Variable)

* `void reset()`
    reset state to the default value

* `void set_state(const Tensor& state)`
    set new value for State

* `Tensor get_state() const`
    returns current value of State
