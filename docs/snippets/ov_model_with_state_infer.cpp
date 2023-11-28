// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "openvino/op/util/variable.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

int main(int argc, char* argv[]) {
    try {
        // --------------------------- 1. Load inference engine -------------------------------------
        std::cout << "Loading OpenVINO" << std::endl;
        ov::Core core;

        //! [model_create]
        auto arg = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1});
        auto init_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{1, 1}, ov::element::f32, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::opset11::ReadValue>(init_const, variable);
        auto add = std::make_shared<ov::opset11::Add>(arg, read);
        auto assign = std::make_shared<ov::opset11::Assign>(add, variable);
        auto add2 = std::make_shared<ov::opset11::Add>(add, read);
        auto res = std::make_shared<ov::opset11::Result>(add2);

        auto model =
            std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg}));
        //! [model_create]

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        std::cout << "Loading network files" << std::endl;

        // 3. Load network to CPU
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        // 4. Create Infer Request
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // 5. Prepare inputs

        std::vector<ov::Tensor> input_tensors;
        for (const auto& input : compiled_model.inputs()) {
            input_tensors.emplace_back(infer_request.get_tensor(input));
        }

        // 6. Prepare outputs
        std::vector<ov::Tensor> output_tensors;
        for (const auto& output : compiled_model.outputs()) {
            output_tensors.emplace_back(infer_request.get_tensor(output));
        }

        // 7. Initialize memory state before starting
        for (auto&& state : infer_request.query_state()) {
            state.reset();
        }

        //! [part1]
        // input data
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        // infer the first utterance
        for (size_t next_input = 0; next_input < data.size() / 2; next_input++) {
            auto minput = input_tensors[0];

            std::memcpy(minput.data(), &data[next_input], sizeof(float));

            infer_request.infer();
            // check states
            auto states = infer_request.query_state();
            if (states.empty()) {
                throw std::runtime_error("Queried states are empty");
            }
            auto mstate = states[0].get_state();
            if (!mstate) {
                throw std::runtime_error("Can't cast state to MemoryBlob");
            }
            float* state = mstate.data<float>();
            std::cout << state[0] << "\n";
        }

        // resetting state between utterances
        std::cout << "Reset state\n";
        for (auto&& state : infer_request.query_state()) {
            state.reset();
        }

        // infer the second utterance
        for (size_t next_input = data.size() / 2; next_input < data.size(); next_input++) {
            auto minput = input_tensors[0];

            std::memcpy(minput.data(), &data[next_input], sizeof(float));

            infer_request.infer();
            // check states
            auto states = infer_request.query_state();
            auto mstate = states[0].get_state();
            float* state = mstate.data<float>();
            std::cout << state[0] << "\n";
        }
        //! [part1]
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened" << std::endl;
        return 1;
    }

    std::cerr << "Execution successful" << std::endl;
    return 0;
}
