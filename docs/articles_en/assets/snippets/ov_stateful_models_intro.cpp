// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <openvino/opsets/opset8.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/pass/low_latency.hpp>
#include <openvino/pass/manager.hpp>
#include "openvino/core/partial_shape.hpp"
#include "openvino/pass/make_stateful.hpp"

using namespace ov;

void state_network_example () {
    //! [ov:stateful_model]
    // ...

    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 1});
    auto init_const = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});

    // Typically ReadValue/Assign operations are presented as pairs in models.
    // ReadValue operation reads information from an internal memory buffer, Assign operation writes data to this buffer.
    // For each pair, its own Variable object must be created.
    // Variable defines name, shape and type of the buffer.
    const std::string variable_name("variable0");
    ov::op::util::VariableInfo var_info = {init_const->get_shape(),
                                           init_const->get_element_type(),
                                           variable_name};
    auto variable = std::make_shared<ov::op::util::Variable>(var_info);

    // Creating ov::Model
    auto read = std::make_shared<ov::opset8::ReadValue>(init_const, variable);
    auto add = std::make_shared<ov::opset8::Add>(input, read);
    auto save = std::make_shared<ov::opset8::Assign>(add, variable);
    auto result = std::make_shared<ov::opset8::Result>(add);

    auto model = std::make_shared<ov::Model>(ov::ResultVector({result}),
                                             ov::SinkVector({save}),
                                             ov::ParameterVector({input}));
    //! [ov:stateful_model]
}

void low_latency_2_example() {
    //! [ov:low_latency_2]
    // Precondition for ov::Model.
    // TensorIterator and Parameter are created in body of TensorIterator with names
    std::string tensor_iterator_name = "TI_name";
    std::string body_parameter_name = "body_parameter_name";
    std::string idx = "0"; // this is a first variable in the network

    // The State will be named "TI_name/param_name/variable_0"
    auto state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx;

    //! [ov:get_ov_model]
    ov::Core core;
    auto ov_model = core.read_model("path_to_the_model");
    //! [ov:get_ov_model]
    // reshape input if needed

    //! [ov:reshape_ov_model]
    ov_model->reshape({{"X", ov::Shape({1, 1, 16})}});
    //! [ov:reshape_ov_model]

    //! [ov:apply_low_latency_2]
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::LowLatency2>();
    manager.run_passes(ov_model);
    //! [ov:apply_low_latency_2]

    auto hd_specific_model = core.compile_model(ov_model);
    // Try to find the Variable by name
    auto infer_request = hd_specific_model.create_infer_request();
    auto states = infer_request.query_state();
    for (auto& state : states) {
        auto name = state.get_name();
        if (name == state_name) {
            // some actions
        }
    }
    //! [ov:low_latency_2]

    //! [ov:low_latency_2_use_parameters]
    manager.register_pass<ov::pass::LowLatency2>(false);
    //! [ov:low_latency_2_use_parameters]
}

void replace_non_reshapable_const() {
    //! [ov:replace_const]
    // OpenVINO example. How to replace a Constant with hardcoded values of shapes in the network with another one with the new values.
    // Assume we know which Constant (const_with_hardcoded_shape) prevents the reshape from being applied.
    // Then we can find this Constant by name on the network and replace it with a new one with the correct shape.
    ov::Core core;
    auto model = core.read_model("path_to_model");
    // Creating the new Constant with a correct shape.
    // For the example shown in the picture above, the new values of the Constant should be 1, 1, 10 instead of 1, 49, 10
    auto new_const = std::make_shared<ov::opset8::Constant>( /*type, shape, value_with_correct_shape*/ );
    for (const auto& node : model->get_ops()) {
        // Trying to find the problematic Constant by name.
        if (node->get_friendly_name() == "name_of_non_reshapable_const") {
            auto const_with_hardcoded_shape = ov::as_type_ptr<ov::opset8::Constant>(node);
            // Replacing the problematic Constant with a new one. Do this for all the problematic Constants in the network, then
            // you can apply the reshape feature.
            ov::replace_node(const_with_hardcoded_shape, new_const);
        }
    }
    //! [ov:replace_const]
}

void apply_make_stateful_tensor_names() {
    //! [ov:make_stateful_tensor_names]
    ov::Core core;
    auto ov_model = core.read_model("path_to_the_model");
    std::map<std::string, std::string> tensor_names = {{"tensor_name_1", "tensor_name_4"},
                                                  {"tensor_name_3", "tensor_name_6"}};
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MakeStateful>(tensor_names);
    manager.run_passes(ov_model);
    //! [ov:make_stateful_tensor_names]
}

void apply_make_stateful_ov_nodes() {
    //! [ov:make_stateful_ov_nodes]
    ov::Core core;
    auto ov_model = core.read_model("path_to_the_model");
    // Parameter_1, Result_1, Parameter_3, Result_3 are shared_ptr<Parameter/Result> in the ov_model
    std::vector<std::pair<std::shared_ptr<ov::opset8::Parameter>, std::shared_ptr<ov::opset8::Result>>> pairs
            = {/*Parameter_1, Result_1, Parameter_3, Result_3*/};
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MakeStateful>(pairs);
    manager.run_passes(ov_model);
    //! [ov:make_stateful_ov_nodes]
}

int main(int argc, char *argv[]) {
    try {
        //! [ov:state_api_usage]
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
    //! [ov:state_api_usage]
    return 0;
}
