---
sidebar_position: 1
sidebar_label: "Stateful models and State API"
---

# Stateful models and State API


A "stateful model" is a model that implicitly preserves data between two consecutive inference
calls. The tensors saved from one run are kept in an internal memory buffer called a
"state" or a "variable" and may be passed to the next run, while never being exposed as model
output. In contrast, for a "stateless" model to pass data between runs, all produced data is
returned as output and needs to be handled by the application itself for reuse at the next
execution.

<div style={{textAlign: 'center'}}>

![example comparison between stateless and stateful model implementations](stateful_model_example.svg)

</div>

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

## OpenVINO Stateful Model Representation

To make a model stateful, OpenVINO replaces looped pairs of Parameter and Result with its
own two operations:

- `ReadValue` ([see specs](../../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/read-value-6.md))
  reads the data from the state and returns it as output.
- `Assign` ([see specs](../../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/assign-6.md))
  accepts the data as input and saves it in the state for the next inference call.

Each pair of these operations works with **state**, which is automatically saved between
inference runs and can be reset when needed. This way, the burden of copying data is shifted
from the application code to OpenVINO and all related internal work is hidden from the user.

There are three methods of turning an OpenVINO model into a stateful one:

- [Optimum-Intel](../../../../openvino-workflow-generative/inference-with-optimum-intel) - the most user-friendly option. All necessary optimizations
  are recognized and applied automatically. The drawback is, the tool does not work with all
  models.
- [MakeStateful transformation](obtaining-stateful-openvino-model.md#ov-ug-make-stateful) - enables the user to choose which
  pairs of Parameter and Result to replace, as long as the paired operations are of the same
  shape and element type.
- [LowLatency2 transformation](obtaining-stateful-openvino-model.md#ov-ug-low-latency) - automatically detects and replaces
  Parameter and Result pairs connected to hidden and cell state inputs of LSTM/RNN/GRU operations
  or Loop/TensorIterator operations.


## Running Inference of Stateful Models {#ov-ug-stateful-model-inference}

For the most basic applications, stateful models work out of the box. For additional control,
OpenVINO offers a dedicated API, whose methods enable you to both retrieve and change data
saved in states between inference runs. OpenVINO runtime uses `ov::InferRequest::query_state`
to get the list of states from a model and the `ov::VariableState` class to operate with
states.

**\`ov::InferRequest\` methods:**  
`std::vector<VariableState> query_state();` - gets all available states for the given
inference request  
`void reset_state()` - resets all States to their default values  
  
**\`ov::VariableState\` methods:**  
`std::string get_name() const` - returns name(variable\_id) of the corresponding
State(Variable)  
`void reset()` - resets the state to the default value  
`void set_state(const Tensor& state)` - sets a new value for the state  
`Tensor get_state() const` - returns the current value of the state

**Using multiple threads**  
Note that if multiple independent sequences are involved, several threads may be used to
process each section in its own infer request. However, using several infer requests
for one sequence is not recommended, as the state would not be passed automatically. Instead,
each run performed in a different infer request than the previous one would require the state
to be set "manually", using the `ov::VariableState::set_state` method.

<div style={{textAlign: 'center'}}>

![diagram of how initial state value is set or reset](stateful_model_init_subgraph.svg)

</div>

**Resetting states**  
Whenever it is necessary to set the initial value of a state or reset it, an initializing  
subgraph for the ReadValue operation and a special `reset` method are provided.  
A case worth mentioning here is, if you decide to reset, query for states, and then retrieve  
state data. It will result in undefined values and so, needs to be avoided.

## Stateful Model Application Example

Here is a code example demonstrating inference of three independent sequences of data.
One infer request and one thread are used. The state should be reset between consecutive
sequences.

**C++**

```cpp
        // 1. Load inference engine
        std::cout << "Loading Inference Engine" << std::endl;
        ov::Core ie;

        // 2. Read a model
        std::cout << "Loading network files" << std::endl;
        std::shared_ptr<Model> network;
        network = ie.read_model("path_to_ir_xml_from_the_previous_section");
        network->get_parameters()[0]->set_layout("NC");
        set_batch(network, 1);

        // 3. Load network to CPU
        CompiledModel hw_specific_model = ie.compile_model(network, "CPU");

        // 4. Create Infer Request
        InferRequest inferRequest = hw_specific_model.create_infer_request();

        // 5. Reset memory states before starting
        auto states = inferRequest.query_state();
        if (states.size() != 1) {
            std::string err_message = "Invalid queried state number. Expected 1, but got "
                                      + std::to_string(states.size());
            throw std::runtime_error(err_message);
        }
        inferRequest.reset_state();

        // 6. Inference
        std::vector<float> input_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        // This example demonstrates how to work with OpenVINO State API.
        // Input_data: some array with 12 float numbers

        // Part1: read the first four elements of the input_data array sequentially.
        // Expected output for the first utterance:
        // sum of the previously processed elements [ 1, 3, 6, 10]

        // Part2: reset state value (set to 0) and read the next four elements.
        // Expected output for the second utterance:
        // sum of the previously processed elements [ 5, 11, 18, 26]

        // Part3: set state value to 5 and read the next four elements.
        // Expected output for the third utterance:
        // sum of the previously processed elements + 5 [ 14, 24, 35, 47]
        auto& target_state = states[0];

        // Part 1
        std::cout << "Infer the first utterance" << std::endl;
        for (size_t next_input = 0; next_input < input_data.size()/3; next_input++) {
            auto in_tensor = inferRequest.get_input_tensor(0);
            std::memcpy(in_tensor.data(), &input_data[next_input], sizeof(float));

            inferRequest.infer();
            auto state_buf = target_state.get_state().data<float>();
            std::cout << state_buf[0] << "\n";
        }

        // Part 2
        std::cout<<"\nReset state between utterances...\n";
        target_state.reset();

        std::cout << "Infer the second utterance" << std::endl;
        for (size_t next_input = input_data.size()/3; next_input < (input_data.size()/3 * 2); next_input++) {
            auto in_tensor = inferRequest.get_input_tensor(0);
            std::memcpy(in_tensor.data(), &input_data[next_input], sizeof(float));

            inferRequest.infer();
            auto state_buf = target_state.get_state().data<float>();
            std::cout << state_buf[0] << "\n";
        }

        // Part 3
        std::cout<<"\nSet state value between utterances to 5...\n";
        std::vector<float> v = {5};
        Tensor tensor(element::f32, Shape{1, 1});
        std::memcpy(tensor.data(), &v[0], sizeof(float));
        target_state.set_state(tensor);

        std::cout << "Infer the third utterance" << std::endl;
        for (size_t next_input = (input_data.size()/3 * 2); next_input < input_data.size(); next_input++) {
            auto in_tensor = inferRequest.get_input_tensor(0);
            std::memcpy(in_tensor.data(), &input_data[next_input], sizeof(float));

            inferRequest.infer();

            auto state_buf = target_state.get_state().data<float>();
            std::cout << state_buf[0] << "\n";
        }

    }
    catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown/internal exception happened" << std::endl;
        return 1;
    }

    std::cout << "Execution successful" << std::endl;
```

You can find more examples demonstrating how to work with states in other articles:

- [LLaVA-NeXT Multimodal Chatbot notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llava-next-multimodal-chatbot)
