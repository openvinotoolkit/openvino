// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/duplicate_inputs_outputs_names.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/runtime/core.hpp"

namespace {

constexpr char DUMMY_NAME[] = "dummy_name";

}  // namespace

namespace ExecutionGraphTests {

std::string ExecGraphDuplicateInputsOutputsNames::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string targetDevice = obj.param;
    return "Dev=" + targetDevice;
}

/**
 * Checks whether running predictions on a model containing duplicate names within its inputs/outputs yields the same
 * result as when using unique names for the same architecture.
 */
TEST_P(ExecGraphDuplicateInputsOutputsNames, CheckOutputsMatch) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const std::string device_name = this->GetParam();
    const ov::element::Type precision = ov::element::f32;
    const ov::Shape shape = {3, 2};
    float input_data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float input_data2[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    const ov::Tensor input_tensor1{precision, shape, input_data1};
    const ov::Tensor input_tensor2{precision, shape, input_data2};

    // A simple graph with 2 inputs and 2 outputs
    auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, shape);
    auto input2 = std::make_shared<ov::op::v0::Parameter>(precision, shape);
    auto sum = std::make_shared<ov::op::v1::Add>(input1, input2);
    auto mul = std::make_shared<ov::op::v1::Multiply>(input1, input2);
    auto output1 = std::make_shared<ov::op::v0::Result>(sum->get_default_output());
    auto output2 = std::make_shared<ov::op::v0::Result>(mul->get_default_output());

    // Set the same name for all inputs/outputs
    input1->set_friendly_name(DUMMY_NAME);
    input2->set_friendly_name(DUMMY_NAME);
    input1->get_output_tensor(0).set_names({DUMMY_NAME});
    input2->get_output_tensor(0).set_names({DUMMY_NAME});

    output1->set_friendly_name(DUMMY_NAME);
    output2->set_friendly_name(DUMMY_NAME);
    output1->get_input_tensor(0).set_names({DUMMY_NAME});
    output2->get_input_tensor(0).set_names({DUMMY_NAME});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{output1, output2},
                                             ov::ParameterVector{input1, input2},
                                             "SimpleNetwork1");

    // Load the plugin, compile the model and run a single prediction
    auto core = ov::Core();
    ov::CompiledModel compiled_model_duplicate_names = core.compile_model(model, device_name);
    ov::InferRequest inference_request_duplicate_names = compiled_model_duplicate_names.create_infer_request();

    inference_request_duplicate_names.set_tensor(compiled_model_duplicate_names.input(0), input_tensor1);
    inference_request_duplicate_names.set_tensor(compiled_model_duplicate_names.input(1), input_tensor2);
    inference_request_duplicate_names.infer();

    const ov::Tensor output_tensor1 =
        inference_request_duplicate_names.get_tensor(compiled_model_duplicate_names.output(0));
    const ov::Tensor output_tensor2 =
        inference_request_duplicate_names.get_tensor(compiled_model_duplicate_names.output(1));
    const float* output_buffer1 = output_tensor1.data<float>();
    const float* output_buffer2 = output_tensor2.data<float>();

    // Rebuild the model using unique names for inputs/outputs
    size_t name_index = 0;
    input1->set_friendly_name(DUMMY_NAME + std::to_string(name_index++));
    input2->set_friendly_name(DUMMY_NAME + std::to_string(name_index++));
    input1->get_output_tensor(0).set_names({DUMMY_NAME + std::to_string(name_index++)});
    input2->get_output_tensor(0).set_names({DUMMY_NAME + std::to_string(name_index++)});

    output1->set_friendly_name(DUMMY_NAME + std::to_string(name_index++));
    output2->set_friendly_name(DUMMY_NAME + std::to_string(name_index++));
    output1->get_input_tensor(0).set_names({DUMMY_NAME + std::to_string(name_index++)});
    output2->get_input_tensor(0).set_names({DUMMY_NAME + std::to_string(name_index)});

    model = std::make_shared<ov::Model>(ov::ResultVector{output1, output2},
                                        ov::ParameterVector{input1, input2},
                                        "SimpleNetwork2");

    // Compile the new model and run a single prediction
    ov::CompiledModel compiled_model_unique_names = core.compile_model(model, device_name);
    ov::InferRequest inference_request_unique_names = compiled_model_unique_names.create_infer_request();

    inference_request_unique_names.set_tensor(input1, input_tensor1);
    inference_request_unique_names.set_tensor(input2, input_tensor2);
    inference_request_unique_names.infer();

    const ov::Tensor reference_tensor1 =
        inference_request_unique_names.get_tensor(compiled_model_unique_names.output(0));
    const ov::Tensor reference_tensor2 =
        inference_request_unique_names.get_tensor(compiled_model_unique_names.output(1));
    const float* reference_buffer1 = reference_tensor1.data<float>();
    const float* reference_buffer2 = reference_tensor2.data<float>();

    // Both models are using the same architecture, thus the results should match
    for (size_t element_index = 0; element_index < shape_size(shape); ++element_index) {
        ASSERT_EQ(output_buffer1[element_index], reference_buffer1[element_index]);
        ASSERT_EQ(output_buffer2[element_index], reference_buffer2[element_index]);
    }
}

}  // namespace ExecutionGraphTests
