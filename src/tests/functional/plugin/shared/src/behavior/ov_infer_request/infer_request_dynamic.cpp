// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "ie_extension.h"
#include <condition_variable>
#include "openvino/core/shape.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "transformations/utils/utils.hpp"
#include <string>
#include <ie_core.hpp>
#include <thread>
#include <base/behavior_test_utils.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"
#include "behavior/ov_infer_request/infer_request_dynamic.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestDynamicTests::getTestCaseName(testing::TestParamInfo<OVInferRequestDynamicParams> obj) {
    std::shared_ptr<Model> func;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(func, inOutShapes, targetDevice, configuration) = obj.param;
    std::ostringstream result;
    result << "function=" << func->get_friendly_name() << "_";
    result << "inOutShape=(";
    for (const auto& inOutShape : inOutShapes) {
        result << "(" << CommonTestUtils::vec2str(inOutShape.first) << "_" << CommonTestUtils::vec2str(inOutShape.second) << ")";
    }
    result << ")_";
    result << "targetDevice=" << targetDevice;
    if (!configuration.empty()) {
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}

void OVInferRequestDynamicTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::tie(function, inOutShapes, targetDevice, configuration) = this->GetParam();
}

void OVInferRequestDynamicTests::TearDown() {
    if (!configuration.empty()) {
        PluginCache::get().reset();
    }
    function.reset();
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithoutSetShape) {
    const std::string tensor_name = "input_tensor";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkBoundWithoutSetShape) {
    const std::string tensor_name = "input_tensor";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 5), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
}


TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithGetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    //OV_ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    EXPECT_NE(0, otensor.get_size()); // output tensor is allocated after infer
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(otensor.get_shape(), refOutShape);
}

TEST_P(OVInferRequestDynamicTests, InferUpperBoundNetworkWithGetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 19), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputname = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    //OV_ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputname));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
}

TEST_P(OVInferRequestDynamicTests, InferFullyDynamicNetworkWithGetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = ov::PartialShape::dynamic();
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor, otensor;
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    //OV_ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputName));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputName));
    ASSERT_EQ(otensor.get_shape(), refOutShape);
}

TEST_P(OVInferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetTensorLower) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(2, 3), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    //ASSERT_THROW(req.infer(), ov::Exception);
}

TEST_P(OVInferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetTensorUpper) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(1, 2), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape({3, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    // ASSERT_THROW(req.infer(), ov::Exception);
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithGetTensor2times) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape(refShape));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    req.wait();
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);

    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape(refShape2));
    ASSERT_EQ(tensor.get_shape(), refShape2);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    req.wait();
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape2);
}


TEST_P(OVInferRequestDynamicTests, GetSameTensor2times) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    OV_ASSERT_NO_THROW(tensor.set_shape(refShape));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(function->inputs().back().get_any_name()));
    ASSERT_EQ(tensor.get_shape(), refShape);
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithSetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor(ov::element::f32, refShape);
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
}

TEST_P(OVInferRequestDynamicTests, InferFullyDynamicNetworkWithSetTensor) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = ov::PartialShape::dynamic();
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor(ov::element::f32, refShape), otensor;
    const std::string outputName = function->outputs().back().get_any_name();
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(otensor = req.get_tensor(outputName));
    ASSERT_EQ(0, otensor.get_size()); // output tensor is not allocated
    ASSERT_EQ(function->output(0).get_element_type(), otensor.get_element_type()); // by it has type
    OV_ASSERT_NO_THROW(req.infer());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_EQ(otensor.get_shape(), refOutShape);
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
    ASSERT_EQ(otensor.get_shape(), refOutShape);
}

TEST_P(OVInferRequestDynamicTests, InferDynamicNetworkWithSetTensor2times) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    const std::string outputName = function->outputs().back().get_any_name();
    // Load ov::Model to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::InferRequest req;
    ov::Tensor tensor(ov::element::f32, refShape);

    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape);

    tensor = ov::Tensor(ov::element::f32, refShape2);
    OV_ASSERT_NO_THROW(req.set_tensor(function->inputs().back().get_any_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape2);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    OV_ASSERT_NO_THROW(tensor = req.get_tensor(outputName));
    ASSERT_EQ(tensor.get_shape(), refOutShape2);
}

TEST_P(OVNotSupportRequestDynamicTests, InferDynamicNotSupported) {
    const std::string tensor_name = "input_tensor";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    OV_ASSERT_NO_THROW(function->reshape(shapes));
    const std::string outputName = function->outputs().back().get_any_name();
    // Load ov::Function to target plugins
    ov::CompiledModel execNet;
    ASSERT_THROW((execNet = ie->compile_model(function, targetDevice, configuration)), ov::Exception);
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
