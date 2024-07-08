// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "io_tensor.hpp"

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"

using namespace ov::auto_plugin::tests;

void InferRequest_IOTensor_Test::SetUp() {
    AutoFuncTests::SetUp();
    std::tie(target_device, property) = this->GetParam();
    auto compiled_model =
        core.compile_model(model_cannot_batch, "AUTO", {ov::device::priorities("MOCK_GPU", "MOCK_CPU")});
    input = compiled_model.input();
    output = compiled_model.output();
}

void InferRequest_IOTensor_Test::TearDown() {
    input = {};
    output = {};
    AutoFuncTests::TearDown();
}

TEST_P(InferRequest_IOTensor_Test, fail_to_set_nullptr_for_input) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ASSERT_THROW(req.set_tensor(input, {}), ov::Exception);
}

TEST_P(InferRequest_IOTensor_Test, fail_to_set_nullptr_for_output) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ASSERT_THROW(req.set_tensor(output, {}), ov::Exception);
}

TEST_P(InferRequest_IOTensor_Test, can_set_and_get_input) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    auto tensor = ov::test::utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(input));

    ASSERT_TRUE(actual_tensor);
    ASSERT_NE(nullptr, actual_tensor.data());
    ASSERT_EQ(tensor.data(), actual_tensor.data());
    ASSERT_EQ(input.get_element_type(), actual_tensor.get_element_type());
    ASSERT_EQ(input.get_shape(), actual_tensor.get_shape());
}

TEST_P(InferRequest_IOTensor_Test, fail_to_set_tensor_with_incorrect_name) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    auto tensor = ov::test::utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    ASSERT_THROW(req.set_tensor("incorrect_input", tensor), ov::Exception);
}

TEST_P(InferRequest_IOTensor_Test, fail_input_set_size_incorrect) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    auto shape = input.get_shape();
    shape[0] *= 2;
    auto tensor = ov::test::utils::create_and_fill_tensor(input.get_element_type(), shape);
    ASSERT_THROW(req.set_tensor(input, tensor), ov::Exception);
}

TEST_P(InferRequest_IOTensor_Test, fail_output_set_size_incorrect) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    auto shape = output.get_shape();
    shape[0] *= 2;
    auto tensor = ov::test::utils::create_and_fill_tensor(output.get_element_type(), shape);
    ASSERT_THROW(req.set_tensor(output, tensor), ov::Exception);
}

TEST_P(InferRequest_IOTensor_Test, second_call_get_input) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(InferRequest_IOTensor_Test, second_call_get_output) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(InferRequest_IOTensor_Test, second_call_get_input_after_async) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(InferRequest_IOTensor_Test, second_call_get_output_after_async) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(InferRequest_IOTensor_Test, can_infer_with_set_tensor) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    auto input_tensor = ov::test::utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
    auto output_tensor = ov::test::utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
    OV_ASSERT_NO_THROW(req.infer());

    auto actual_input_tensor = req.get_tensor(input);
    ASSERT_EQ(actual_input_tensor.data(), input_tensor.data());
    auto actual_output_tensor = req.get_tensor(output);
    ASSERT_EQ(actual_output_tensor.data(), output_tensor.data());
}

TEST_P(InferRequest_IOTensor_Test, can_infer_after_io_realloc) {
    auto compiled_model = core.compile_model(model_cannot_batch, target_device, property);
    req = compiled_model.create_infer_request();
    ov::Tensor input_tensor, output_tensor;
    auto in_shape = input.get_shape();
    auto out_shape = output.get_shape();

    // imitates blob reallocation
    OV_ASSERT_NO_THROW(input_tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(input_tensor.set_shape({5, 5, 5, 5}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape(in_shape));

    OV_ASSERT_NO_THROW(output_tensor = req.get_tensor(output));
    OV_ASSERT_NO_THROW(output_tensor.set_shape({20, 20, 20, 20}));
    OV_ASSERT_NO_THROW(output_tensor.set_shape(out_shape));

    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.get_tensor(output));
}
namespace {
auto props = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities("MOCK_GPU", "MOCK_CPU")},
                                   {ov::device::priorities("MOCK_GPU")},
                                   {ov::device::priorities("MOCK_CPU", "MOCK_GPU")}};
};

INSTANTIATE_TEST_SUITE_P(AutoFuncTests,
                         InferRequest_IOTensor_Test,
                         ::testing::Combine(::testing::Values("AUTO"), ::testing::ValuesIn(props())),
                         InferRequest_IOTensor_Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(AutoFuncTestsCumu,
                         InferRequest_IOTensor_Test,
                         ::testing::Combine(::testing::Values("MULTI"), ::testing::ValuesIn(props())),
                         InferRequest_IOTensor_Test::getTestCaseName);
}  // namespace