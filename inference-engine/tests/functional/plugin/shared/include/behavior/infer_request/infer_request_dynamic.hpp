// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

// TODO [mandrono]: move current test case inside CPU plug-in and return the original tests
namespace BehaviorTestsDefinitions {

typedef std::tuple<
        std::shared_ptr<ov::Function>,                                     // ov function
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>,  // input/expected output shapes per inference
        std::string,                                                       // Device name
        std::map<std::string, std::string>                                 // Config
> InferRequestDynamicParams;

class InferRequestDynamicTests : public testing::WithParamInterface<InferRequestDynamicParams>,
                                 public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestDynamicParams> obj) {
        std::shared_ptr<ov::Function> func;
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
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
                result << "configItem=" << configItem.first << "_" << configItem.second << "_";
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::tie(function, inOutShapes, targetDevice, configuration) = this->GetParam();
        ie = ov::test::PluginCache::get().core(targetDevice);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::PluginCache::get().reset();
        }
    }

    std::shared_ptr<ov::runtime::Core> ie = ov::test::PluginCache::get().core();
    std::shared_ptr<ov::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
};

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithoutSetShape) {
    const std::string tensor_name = "Tensor_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkBoundWithoutSetShape) {
    const std::string tensor_name = "Tensor_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 5), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
}


TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithGetTensor) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    //ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    req.wait();
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
}

TEST_P(InferRequestDynamicTests, InferUpperBoundNetworkWithGetTensor) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(0, 19), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    //ASSERT_NO_THROW(req.SetShape(tensor_name, {1, 4, 20, 20}));
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    ASSERT_EQ(tensor.get_shape(), refShape);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    req.wait();
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
}

TEST_P(InferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetTensorLower) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(2, 3), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape({1, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    //ASSERT_THROW(req.infer(), ov::Exception);
}

TEST_P(InferRequestDynamicTests, InferOutOfRangeShapeNetworkWithGetTensorUpper) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension(1, 2), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape({3, 4, 20, 20}));
    // Plugin may or may not throw in case if input tensor has dimensions that are out of bounds
    // ASSERT_THROW(req.infer(), ov::Exception);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithGetTensor2times) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape(refShape));
    ASSERT_EQ(tensor.get_shape(), refShape);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    req.wait();
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape);

    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape(refShape2));
    ASSERT_EQ(tensor.get_shape(), refShape2);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    req.wait();
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape2);
}


TEST_P(InferRequestDynamicTests, GetSameTensor2times) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor;
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_NO_THROW(tensor.set_shape(refShape));
    ASSERT_EQ(tensor.get_shape(), refShape);
    ASSERT_NO_THROW(tensor = req.get_tensor(function->get_parameters().back()->get_friendly_name()));
    ASSERT_EQ(tensor.get_shape(), refShape);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithSetTensor) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor(ov::element::f32, refShape);
    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(req.set_tensor(function->get_parameters().back()->get_friendly_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    ASSERT_NO_THROW(req.wait());
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape);
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkWithSetTensor2times) {
    const std::string tensor_name = "Tensor_1";
    const ov::Shape refShape = inOutShapes[0].first;
    const ov::Shape refShape2 = inOutShapes[1].first;
    const ov::Shape refOutShape = inOutShapes[0].second;
    const ov::Shape refOutShape2 = inOutShapes[1].second;
    std::map<std::string, ov::PartialShape> shapes;
    shapes[tensor_name] = {ov::Dimension::dynamic(), 4, 20, 20};
    ASSERT_NO_THROW(function->reshape(shapes));
    // Load ov::Function to target plugins
    auto execNet = ie->compile_model(function, targetDevice, configuration);
    // Create InferRequest
    ov::runtime::InferRequest req;
    ov::runtime::Tensor tensor(ov::element::f32, refShape);

    ASSERT_NO_THROW(req = execNet.create_infer_request());
    ASSERT_NO_THROW(req.set_tensor(function->get_parameters().back()->get_friendly_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    ASSERT_NO_THROW(req.wait());
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape);

    tensor = ov::runtime::Tensor(ov::element::f32, refShape2);
    ASSERT_NO_THROW(req.set_tensor(function->get_parameters().back()->get_friendly_name(), tensor));
    ASSERT_EQ(tensor.get_shape(), refShape2);
    ASSERT_NO_THROW(req.infer());
    ASSERT_NO_THROW(req.start_async());
    ASSERT_NO_THROW(req.wait());
    ASSERT_NO_THROW(tensor = req.get_tensor(ngraph::op::util::create_ie_output_name(function->get_results().front()->input_value(0))));
    ASSERT_EQ(tensor.get_shape(), refOutShape2);
}

}  // namespace BehaviorTestsDefinitions