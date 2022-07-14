// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>
#include <future>

#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include "behavior/ov_infer_request/io_tensor.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/op/parameter.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestIOTensorTest::getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

void OVInferRequestIOTensorTest::SetUp() {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    OVInferRequestTests::SetUp();
    req = execNet.create_infer_request();
    input = execNet.input();
    output = execNet.output();
}

void OVInferRequestIOTensorTest::TearDown() {
    req = {};
    input = {};
    output = {};
    OVInferRequestTests::TearDown();
}

TEST_P(OVInferRequestIOTensorTest, Cancreate_infer_request) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
}

TEST_P(OVInferRequestIOTensorTest, failToSetNullptrForInput) {
    ASSERT_THROW(req.set_tensor(input, {}), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, failToSetNullptrForOutput) {
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_THROW(req.set_tensor(output, {}), ov::Exception);
}

TEST_P(OVInferRequestIOTensorTest, getAfterSetInputDoNotChangeInput) {
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

TEST_P(OVInferRequestIOTensorTest, getAfterSetInputDoNotChangeOutput) {
    auto tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    req.set_tensor(output, tensor);
    auto actual_tensor = req.get_tensor(output);

    ASSERT_TRUE(actual_tensor);
    ASSERT_FALSE(actual_tensor.data() == nullptr);
    ASSERT_EQ(actual_tensor.data(), tensor.data());
    ASSERT_EQ(output.get_element_type(), actual_tensor.get_element_type());
    ASSERT_EQ(output.get_shape(), actual_tensor.get_shape());
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
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(input));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(input));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTest, secondCallGetOutputAfterInferSync) {
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ov::Tensor tensor1, tensor2;
    OV_ASSERT_NO_THROW(tensor1 = req.get_tensor(output));
    OV_ASSERT_NO_THROW(tensor2 = req.get_tensor(output));
    ASSERT_EQ(tensor1.data(), tensor2.data());
}

TEST_P(OVInferRequestIOTensorTest, canSetInputTensorForInferRequest) {
    auto input_tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(input));
    ASSERT_EQ(input_tensor.data(), actual_tensor.data());
}

TEST_P(OVInferRequestIOTensorTest, canSetOutputBlobForInferRequest) {
    auto output_tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
    ov::Tensor actual_tensor;
    OV_ASSERT_NO_THROW(actual_tensor = req.get_tensor(output));
    ASSERT_EQ(output_tensor.data(), actual_tensor.data());
}

TEST_P(OVInferRequestIOTensorTest, canInferWithSetInOutBlobs) {
    auto input_tensor = utils::create_and_fill_tensor(input.get_element_type(), input.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(input, input_tensor));
    auto output_tensor = utils::create_and_fill_tensor(output.get_element_type(), output.get_shape());
    OV_ASSERT_NO_THROW(req.set_tensor(output, output_tensor));
    OV_ASSERT_NO_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTest, canInferWithGetIn) {
    ov::Tensor input_tensor;
    OV_ASSERT_NO_THROW(input_tensor = req.get_tensor(input));
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

TEST_P(OVInferRequestIOTensorTest, InferStaticNetworkSetInputTensor) {
    const ov::Shape shape1 = {1, 1, 32, 32};
    const ov::Shape shape2 = {1, 1, 40, 40};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[function->inputs().back().get_any_name()] = shape1;
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
    // Get input_tensor
    ov::runtime::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    // Set shape
    OV_ASSERT_NO_THROW(tensor.set_shape(shape2));
    ASSERT_ANY_THROW(req.infer());
}

TEST_P(OVInferRequestIOTensorTest, InferStaticNetworkSetOutputTensor) {
    const ov::Shape shape1 = {1, 1, 32, 32};
    const ov::Shape shape2 = {1, 20};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[function->inputs().back().get_any_name()] = shape1;
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    std::shared_ptr<ov::Core> ie = utils::PluginCache::get().core();
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.infer());
    // Get output_tensor
    ov::runtime::Tensor tensor;
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->outputs().back().get_any_name()););
    // Set shape
    OV_ASSERT_NO_THROW(tensor.set_shape(shape2));
    ASSERT_ANY_THROW(req.infer());
}

std::string OVInferRequestIOTensorSetPrecisionTest::getTestCaseName(const testing::TestParamInfo<OVInferRequestSetPrecisionParams>& obj) {
    element::Type type;
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(type, targetDevice, configuration) = obj.param;
    std::ostringstream result;
    result << "type=" << type << "_";
    result << "targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        using namespace CommonTestUtils;
        for (auto &configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}

void OVInferRequestIOTensorSetPrecisionTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::tie(element_type, target_device, config) = this->GetParam();
    function = ngraph::builder::subgraph::makeConvPoolRelu();
    execNet = core->compile_model(function, target_device, config);
    req = execNet.create_infer_request();
}

void OVInferRequestIOTensorSetPrecisionTest::TearDown() {
    execNet = {};
    req = {};
    auto &apiSummary = ov::test::utils::ApiSummary::getInstance();
    if (this->HasFailure()) {
        apiSummary.updateStat(ov::test::utils::ov_entity::ov_infer_request, target_device, ov::test::utils::PassRate::Statuses::FAILED);
    } else if (this->IsSkipped()) {
        apiSummary.updateStat(ov::test::utils::ov_entity::ov_infer_request, target_device, ov::test::utils::PassRate::Statuses::SKIPPED);
    } else {
        apiSummary.updateStat(ov::test::utils::ov_entity::ov_infer_request, target_device, ov::test::utils::PassRate::Statuses::PASSED);
    }
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
    std::string targetDevice;
    AnyMap configuration;
    std::tie(type, targetDevice, configuration) = obj.param;
    std::ostringstream result;
    result << "type=" << type << "_";
    result << "targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        using namespace CommonTestUtils;
        for (auto &configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}

void OVInferRequestCheckTensorPrecision::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::tie(element_type, target_device, config) = this->GetParam();
    APIBaseTest::SetUp();
    {
        auto parameter1 = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1, 3, 2, 2});
        auto parameter2 = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1, 3, 2, 2});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{parameter1, parameter2}, 1);
        auto result = std::make_shared<ov::op::v0::Result>(concat);
        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter1, parameter2});
    }
    compModel = core->compile_model(model, target_device, config);
    req = compModel.create_infer_request();
}

void OVInferRequestCheckTensorPrecision::TearDown() {
    compModel = {};
    req = {};
    APIBaseTest::TearDown();
}

void OVInferRequestCheckTensorPrecision::Run() {
    EXPECT_EQ(element_type, compModel.input(0).get_element_type());
    EXPECT_EQ(element_type, compModel.input(1).get_element_type());
    EXPECT_EQ(element_type, compModel.output().get_element_type());
    EXPECT_EQ(element_type, req.get_input_tensor(0).get_element_type());
    EXPECT_EQ(element_type, req.get_input_tensor(1).get_element_type());
    EXPECT_EQ(element_type, req.get_output_tensor().get_element_type());
}

TEST_P(OVInferRequestCheckTensorPrecision, CheckInputsOutputs) {
    Run();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
