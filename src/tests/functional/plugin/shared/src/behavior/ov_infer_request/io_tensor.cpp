// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>
#include <future>

#include "behavior/ov_infer_request/io_tensor.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/op/parameter.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/result.hpp"
#include "common_test_utils/subgraph_builders/multiple_input_outpput_double_concat.hpp"
#include "common_test_utils/subgraph_builders/single_split.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"

namespace ov {
namespace test {
namespace behavior {

void OVInferRequestIOTensorTest::SetUp() {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OVInferRequestTests::SetUp();
    try {
        req = execNet.create_infer_request();
    } catch (const std::exception& ex) {
        FAIL() << "Can't Create Infer Requiest in SetUp \nException [" << ex.what() << "]"
               << std::endl;
    }
    input = execNet.input();
    output = execNet.output();
}

void OVInferRequestIOTensorTest::TearDown() {
    req = {};
    input = {};
    output = {};
    OVInferRequestTests::TearDown();
}

TEST_P(OVInferRequestIOTensorTest, failToSetNullptrForInput) {
    ASSERT_THROW(req.set_tensor(input, {}), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, failToSetNullptrForOutput) {
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_THROW(req.set_tensor(output, {}), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, failToSetUninitializedInputTensor) {
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_THROW(req.set_tensor(input, tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, failToSetUninitializedOutputTensor) {
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_THROW(req.set_tensor(output, tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, canSetAndGetInput) {
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(input));

    ASSERT_TRUE(actual_tensor);
    ASSERT_NE(nullptr, actual_tensor.data());
    ASSERT_EQ(tensor.data(), actual_tensor.data());
    ASSERT_EQ(input.get_element_type(), actual_tensor.get_element_type());
    ASSERT_EQ(input.get_shape(), actual_tensor.get_shape());
}

TEST_P(OVInferRequestIOTensorTest, canSetAndGetOutput) {
    auto tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    req.set_tensor(output, tensor);
    auto actual_tensor = req.get_tensor(output);

    ASSERT_TRUE(actual_tensor);
    ASSERT_FALSE(actual_tensor.data() == nullptr);
    ASSERT_EQ(actual_tensor.data(), tensor.data());
    ASSERT_EQ(output.get_element_type(), actual_tensor.get_element_type());
    ASSERT_EQ(output.get_shape(), actual_tensor.get_shape());
}


TEST_P(OVInferRequestIOTensorTest, getAfterSetInputDoNotChangeInput) {
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(input));

    ASSERT_EQ(tensor.data(), actual_tensor.data());
    ASSERT_EQ(tensor.get_shape(), actual_tensor.get_shape());
    ASSERT_EQ(tensor.get_element_type(), actual_tensor.get_element_type());
}

TEST_P(OVInferRequestIOTensorTest, getAfterSetOutputDoNotChangeOutput) {
    auto tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(output, tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(output));

    ASSERT_EQ(tensor.data(), actual_tensor.data());
    ASSERT_EQ(tensor.get_shape(), actual_tensor.get_shape());
    ASSERT_EQ(tensor.get_element_type(), actual_tensor.get_element_type());
}

TEST_P(OVInferRequestIOTensorTest, failToSetTensorWithIncorrectName) {
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    ASSERT_THROW(req.set_tensor("incorrect_input", tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, failToSetInputWithIncorrectSizes) {
    auto shape = input.get_shape();
    shape[0] *= 2;
    auto tensor = utils::create_and_fill_tensor(input.get_element_type(), shape);
    ASSERT_THROW(req.set_tensor(input, tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, failToSetOutputWithIncorrectSizes) {
    auto shape = output.get_shape();
    shape[0] *= 2;
    auto tensor = utils::create_and_fill_tensor(output.get_element_type(), shape);
    ASSERT_THROW(req.set_tensor(output, tensor), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, canInferWithoutSetAndGetInOutSync) {
    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTest, canInferWithoutSetAndGetInOutAsync) {
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestIOTensorTest, secondCallGetInputDoNotReAllocateData) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTest, secondCallGetOutputDoNotReAllocateData) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTest, secondCallGetInputAfterInferSync) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTest, secondCallGetOutputAfterInferSync) {
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTest, canInferWithSetInOutBlobs) {
    auto input_tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
    auto output_tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
    OV_ASSERT_NO_THROW(req.infer());

    auto actual_input_tensor = req.get_tensor(input);
    ASSERT_EQ(actual_input_tensor.data(), input_tensor.data());
    auto actual_output_tensor = req.get_tensor(output);
    ASSERT_EQ(actual_output_tensor.data(), output_tensor.data());
}

TEST_P(OVInferRequestIOTensorTest, canInferWithGetIn) {
    ov::Tensor input_tensor;
    OV_ASSERT_NO_THROW(input_tensor = req.get_tensor(input));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.get_tensor(output));
}

TEST_P(OVInferRequestIOTensorTest, canInferAfterIOBlobReallocation) {
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

TEST_P(OVInferRequestIOTensorTest, canInferWithGetOut) {
    ov::Tensor output_tensor;
    OV_ASSERT_NO_THROW(output_tensor = req.get_tensor(output));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(req.get_tensor(output));
}

TEST_P(OVInferRequestIOTensorTest, InferStaticNetworkSetChangedInputTensorThrow) {
    const ov::Shape shape1 = {1, 2, 40, 40};
    const ov::Shape shape2 = {1, 2, 32, 32};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[function->inputs().back().get_any_name()] = shape1;
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
    // Get input_tensor
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    // Set shape
    OV_ASSERT_NO_THROW(tensor.set_shape(shape2));
    ASSERT_ANY_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTest, InferStaticNetworkSetChangedOutputTensorThrow) {
    const ov::Shape shape1 = {1, 2, 32, 32};
    ov::Shape shape2;
    shape2 = ov::Shape{1, 4, 20, 20};

    std::map<std::string, ov::PartialShape> shapes;
    shapes[function->inputs().back().get_any_name()] = shape1;
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    auto execNet = ie->compile_model(function, target_device, configuration);
    // Create InferRequest
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
    // Get output_tensor
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->outputs().back().get_any_name()););
    // Set shape
    OV_ASSERT_NO_THROW(tensor.set_shape(shape2));
    ASSERT_ANY_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTest, CheckInferIsNotChangeInput) {
    ov::Tensor input_tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
    OV_ASSERT_NO_THROW(req.get_tensor(input));

    OV_ASSERT_NO_THROW(req.infer());

    ov::Tensor input_after_infer;
    OV_ASSERT_NO_THROW(input_after_infer = req.get_tensor(input));
    ov::test::utils::compare(input_tensor, input_after_infer);

    OV_ASSERT_NO_THROW(req.infer());

    ov::Tensor input_after_several_infer;
    OV_ASSERT_NO_THROW(input_after_several_infer = req.get_tensor(input));
    ov::test::utils::compare(input_tensor, input_after_several_infer);
}

std::string OVInferRequestIOTensorSetPrecisionTest::getTestCaseName(const testing::TestParamInfo<OVInferRequestSetPrecisionParams>& obj) {
    element::Type type;
    std::string target_device;
    ov::AnyMap configuration;
    std::tie(type, target_device, configuration) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "type=" << type << "_";
    result << "target_device=" << target_device << "_";
    if (!configuration.empty()) {
        using namespace ov::test::utils;
        for (auto &configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}

void OVInferRequestIOTensorSetPrecisionTest::SetUp() {
    std::tie(element_type, target_device, config) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
    function = ov::test::utils::make_split_concat();
    execNet = core->compile_model(function, target_device, config);
    req = execNet.create_infer_request();
}

void OVInferRequestIOTensorSetPrecisionTest::TearDown() {
    execNet = {};
    req = {};
    APIBaseTest::TearDown();
}

TEST_P(OVInferRequestIOTensorSetPrecisionTest, CanSetInBlobWithDifferentPrecision) {
    for (auto&& output : execNet.outputs()) {
        auto output_tensor = utils::create_and_fill_tensor(element_type, output.get_shape());
        if (output.get_element_type() == element_type) {
            OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
        } else {
            ASSERT_THROW(req.set_tensor(output, output_tensor), ov::Exception);
        }
    }
}

TEST_P(OVInferRequestIOTensorSetPrecisionTest, CanSetOutBlobWithDifferentPrecision) {
    for (auto&& input : execNet.inputs()) {
        auto input_tensor = utils::create_and_fill_tensor(element_type, input.get_shape());
        if (input.get_element_type() == element_type) {
            OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
        } else {
            ASSERT_THROW(req.set_tensor(input, input_tensor), ov::Exception);
        }
    }
}

std::string OVInferRequestCheckTensorPrecision::getTestCaseName(const testing::TestParamInfo<OVInferRequestCheckTensorPrecisionParams>& obj) {
    element::Type type;
    std::string target_device;
    AnyMap configuration;
    std::tie(type, target_device, configuration) = obj.param;
    std::ostringstream result;
    result << "type=" << type << "_";
    result << "target_device=" << target_device << "_";
    if (!configuration.empty()) {
        using namespace ov::test::utils;
        for (auto &configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}

void OVInferRequestCheckTensorPrecision::SetUp() {
    std::tie(element_type, target_device, config) = this->GetParam();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    APIBaseTest::SetUp();
}

bool OVInferRequestCheckTensorPrecision::compareTensors(const ov::Tensor& t1, const ov::Tensor& t2) {
    void* data1;
    void* data2;
    try {
        data1 = t1.data();
    } catch (const ov::Exception&) {
        // Remote tensor
        data1 = nullptr;
    }
    try {
        data2 = t2.data();
    } catch (const ov::Exception&) {
        // Remote tensor
        data2 = nullptr;
    }
    const auto strides_eq = (t1.get_element_type().bitwidth() >= 8 && t2.get_element_type().bitwidth() >= 8) ?  t1.get_strides() == t2.get_strides() : true;
    return t1.get_element_type() == t2.get_element_type() && t1.get_shape() == t2.get_shape() &&
               t1.get_byte_size() == t2.get_byte_size() && t1.get_size() == t2.get_size() && strides_eq && data1 == data2;
}

void OVInferRequestCheckTensorPrecision::createInferRequest() {
    try {
        compModel = core->compile_model(model, target_device, config);
        request = compModel.create_infer_request();
    } catch (std::runtime_error& e) {
        const std::string errorMsg = e.what();
        const auto expectedMsg = exp_error_str_;
        ASSERT_STR_CONTAINS(errorMsg, expectedMsg);
        EXPECT_TRUE(errorMsg.find(expectedMsg) != std::string::npos)
            << "Wrong error message, actual error message: " << errorMsg << ", expected: " << expectedMsg;
        if (std::count(precisions.begin(), precisions.end(), element_type) == 0) {
            GTEST_SKIP_(expectedMsg.c_str());
        } else {
            FAIL() << "Precision " << element_type.c_type_string()
                    << " is marked as supported but the network was not loaded";
        }
    }
}

void OVInferRequestCheckTensorPrecision::TearDown() {
    APIBaseTest::TearDown();
}

TEST_P(OVInferRequestCheckTensorPrecision, getInputFromFunctionWithSingleInput) {
    model = ov::test::utils::make_split_concat({1, 4, 24, 24}, element_type);
    createInferRequest();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.input()));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->input()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.input().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->input().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_input_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVInferRequestCheckTensorPrecision, getOutputFromFunctionWithSingleInput) {
    model = ov::test::utils::make_split_concat({1, 4, 24, 24}, element_type);
    createInferRequest();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.output()));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.output().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output().get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVInferRequestCheckTensorPrecision, getInputsFromFunctionWithSeveralInputs) {
    model = ov::test::utils::make_multiple_input_output_double_concat({1, 1, 32, 32}, element_type);
    createInferRequest();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.input(0)));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->input(0)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.input(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->input(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_input_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.input(1)));
    try {
        // To avoid case with remote tensors
        tensor1.data();
        EXPECT_FALSE(compareTensors(tensor1, tensor2));
    } catch (const ov::Exception&) {
    }
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->input(1)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.input(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->input(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_input_tensor(1));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVInferRequestCheckTensorPrecision, getOutputsFromFunctionWithSeveralOutputs) {
    model = ov::test::utils::make_multiple_input_output_double_concat({1, 1, 32, 32}, element_type);
    createInferRequest();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.output(0)));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(0)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.output(1)));
    try {
        // To avoid case with remote tensors
        tensor1.data();
        EXPECT_FALSE(compareTensors(tensor1, tensor2));
    } catch (const ov::Exception&) {
    }
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(1)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(1));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

TEST_P(OVInferRequestCheckTensorPrecision, getOutputsFromSplitFunctionWithSeveralOutputs) {
    model = ov::test::utils::make_single_split({1, 4, 24, 24}, element_type);
    createInferRequest();

    ov::Tensor tensor1, tensor2;
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.output(0)));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(0)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(0).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(0));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor1 = request.get_tensor(compModel.output(1)));
    try {
        // To avoid case with remote tensors
        tensor1.data();
        EXPECT_FALSE(compareTensors(tensor1, tensor2));
    } catch (const ov::Exception&) {
    }
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(1)));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(compModel.output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_tensor(model->output(1).get_any_name()));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
    EXPECT_NO_THROW(tensor2 = request.get_output_tensor(1));
    EXPECT_TRUE(compareTensors(tensor1, tensor2));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
