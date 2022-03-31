Stateful models and State API {#openvino_docs_OV_UG_stateful_models_intro}
==============================

@sphinxdirective

.. toctree::
    :maxdepth: 1
    :hidden:

    openvino_docs_OV_UG_ways_to_get_stateful_model

@endsphinxdirective

## What is a Stateful Network?

 Several use cases require processing of data sequences. When length of a sequence is known and small enough, 
 we can process it with RNN like networks that contain a cycle inside. But in some cases, like online speech recognition of time series 
 forecasting, length of data sequence is unknown. Then data can be divided in small portions and processed step-by-step. But dependency 
 between data portions should be addressed. For that, networks save some data between inferences - state. When one dependent sequence is over,
 state should be reset to initial value and new sequence can be started.
 
Deep learning frameworks provide a dedicated API to build models with state. For example, Keras has special option for RNNs `stateful` that turns on saving state 
 between inferences. Kaldi contains special specifier `Offset` to define time offset in a network. 
 
 OpenVINO also contains special API to simplify work with networks with states. State is automatically saved between inferences, 
 and there is a way to reset state when needed. You can also read state or set it to some new value between inferences.
 
## OpenVINO State Representation

 OpenVINO contains a special abstraction `Variable` to represent a state in a network. There are two operations to work with the state: 
* `Assign` to save value in state
* `ReadValue` to read value saved on previous iteration

![state_network_example](./img/state_network_example.png)

You can find more details on these operations in [ReadValue specification](../ops/infrastructure/ReadValue_3.md) and 
[Assign specification](../ops/infrastructure/Assign_3.md).

## How to get the OpenVINO Model with states

* [Convert Kaldi model to IR via Model Optimizer.](../MO_DG/prepare_model/convert_model/kaldi_specific)
   If the original Kaldi model contains RNN-like operations with `stateful` option, then after ModelOptimizer conversion,
   the resulting OpenVINO model will also contain states.

* [Apply LowLatency2 transformation.](./ways_to_get_stateful_model.md#)
   If a model contains a loop that runs over some sequence of input data,
   the LowLatency2 transformation can be applied to get model with states.
   Note: there are some [specific limitations]() to use the transformation.

* [Apply MakeStateful transformation.](./ways_to_get_stateful_model.md)
   If after conversion from original model to OpenVINO representation, the resulting model contains Parameter and Result operations,
   which pairwise have the same shape and element type, the MakeStateful transformation can be applied to get model with states.

* [Create the model via OpenVINO API.](./ways_to_get_stateful_model.md)
   For testing purposes or for some specific cases, when the ways to get OpenVINO model with states described above are not enough for your purposes,
   you can use OpenVINO API and create `ov::opset8::ReadValue` and `ov::opset8::Assign` operations directly.

## OpenVINO State API

OpenVINO runtime has the `ov::InferRequest::query_state` method  to get the list of states from a network and `ov::VariableState` class to operate with states. 
 Below you can find brief description of methods and the example of how to use this interface.
 
 * `std::string get_name() const`
   returns name(variable_id) of according Variable
 * `void reset()`
   reset state to default value
 * `void set_state(const Tensor& state)`
   set new value for state
 * `Tensor get_state() const`
   returns current value of state

## Example of Stateful Network Inference

The example below demonstrates inference of three independent sequences of data. State should be reset between these sequences.

One infer request and one thread will be used in this example. Using several threads is possible if you have several independent sequences. Then each sequence can be processed in its own infer request. Inference of one sequence in several infer requests is not recommended. In one infer request state will be saved automatically between inferences, but 
if the first step is done in one infer request and the second in another, state should be set in new infer request manually (using `ov::VariableState::set_state` method).

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_network_state_intro.cpp
         :language: cpp
         :fragment: [ov:state_api_usage]

@endsphinxdirective

You can find more powerful examples demonstrating how to work with networks with states in speech sample and demo. 
Descriptions can be found in [Samples Overview](./Samples_Overview.md)
