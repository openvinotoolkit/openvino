// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <common_test_utils/test_assertions.hpp>
#include <sstream>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> getConstantGraph(element::Type type) {
    ResultVector results;
    ParameterVector params;
    auto op = std::make_shared<ov::op::v1::Add>(opset8::Constant::create(type, {1}, {1}),
                                                opset8::Constant::create(type, {1}, {1}));
    op->set_friendly_name("Add");
    auto res = std::make_shared<ov::op::v0::Result>(op);
    res->set_friendly_name("Result");
    res->get_output_tensor(0).set_names({"tensor_output"});
    results.push_back(res);
    return std::make_shared<Model>(results, params);
}

inline std::shared_ptr<ov::Model> createSimpleModel() {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto add_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(data, add_constant);

    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{data});
}

inline std::shared_ptr<ov::Model> createModelContainingSubgraph() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    input->get_output_tensor(0).set_names({"tensor_input"});

    // Simple subgraph that adds 1 to the input
    std::shared_ptr<ov::Model> body = createSimpleModel();
    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    tensor_iterator->set_sliced_input(
        std::dynamic_pointer_cast<ov::op::v0::Parameter>(body->input().get_node_shared_ptr()),
        input,
        0,
        1,
        1,
        -1,
        0);

    // Outside the subgraph, add 1 to the output
    auto add_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(tensor_iterator->get_iter_value(body->output(), -1), add_constant);

    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->get_output_tensor(0).set_names({"tensor_output"});

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

inline bool isCommandQueueExtSupported() {
    return std::make_shared<::intel_npu::ZeroInitStructsHolder>()->getCommandQueueDdiTable().version() > 0;
}

typedef std::tuple<std::shared_ptr<ov::Model>,  // Model
                   std::string,                 // Device name
                   ov::AnyMap                   // Config
                   >
    CompileAndInferRequestParams;

class OVCompileAndInferRequest : public testing::WithParamInterface<CompileAndInferRequestParams>,
                                 public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompileAndInferRequestParams>& obj) {
        std::shared_ptr<ov::Model> model;
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(model, targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }
    void SetUp() override {
        std::tie(function, target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabled_test_patterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
    }

protected:
    ov::CompiledModel execNet;
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::string targetDevice;
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
};

TEST_P(OVCompileAndInferRequest, AsyncInferRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(exception_ptr, nullptr);
        is_called = true;
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_TRUE(is_called);
}

TEST_P(OVCompileAndInferRequest, PluginWorkloadType) {
    configuration[workload_type.name()] = WorkloadType::DEFAULT;
    auto supportedProperties = core->get_property("NPU", supported_properties.name()).as<std::vector<PropertyName>>();
    bool workloadTypeSupported =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == workload_type.name();
        });

    if (isCommandQueueExtSupported()) {
        ASSERT_TRUE(workloadTypeSupported);
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

        const auto properties = execNet.get_property(supported_properties.name()).as<std::vector<PropertyName>>();
        ASSERT_TRUE(std::any_of(properties.begin(), properties.end(), [](const PropertyName& property) {
            return property == workload_type.name();
        }));

        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        ASSERT_FALSE(workloadTypeSupported);
        OV_EXPECT_THROW_HAS_SUBSTRING(
            core->compile_model(function, target_device, configuration),
            ov::Exception,
            "[ NOT_FOUND ] Option 'WORKLOAD_TYPE' is not supported for current configuration");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadType) {
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::AnyMap modelConfiguration;
    modelConfiguration[workload_type.name()] = WorkloadType::DEFAULT;
    auto supportedProperties = execNet.get_property(supported_properties.name()).as<std::vector<PropertyName>>();
    bool workloadTypeSupported =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == workload_type.name();
        });

    if (isCommandQueueExtSupported()) {
        ASSERT_TRUE(workloadTypeSupported);
        OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        ASSERT_FALSE(workloadTypeSupported);
        OV_EXPECT_THROW_HAS_SUBSTRING(execNet.set_property(modelConfiguration),
                                      ov::Exception,
                                      "Unsupported configuration key: WORKLOAD_TYPE");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadTypeDelayedExecutor) {
    configuration[intel_npu::defer_weights_load.name()] = true;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::AnyMap modelConfiguration;
    modelConfiguration[workload_type.name()] = WorkloadType::DEFAULT;

    if (isCommandQueueExtSupported()) {
        OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        OV_EXPECT_THROW_HAS_SUBSTRING(execNet.set_property(modelConfiguration),
                                      ov::Exception,
                                      "Unsupported configuration key: WORKLOAD_TYPE");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadTypeUpdateAfterCompilation) {
    if (isCommandQueueExtSupported()) {
        configuration[workload_type.name()] = WorkloadType::DEFAULT;
        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));

        ASSERT_EQ(execNet.get_property(workload_type.name()).as<WorkloadType>(), WorkloadType::DEFAULT);
        ov::AnyMap modelConfiguration;
        modelConfiguration[workload_type.name()] = WorkloadType::EFFICIENT;
        OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));
        ASSERT_EQ(execNet.get_property(workload_type.name()).as<WorkloadType>(), WorkloadType::EFFICIENT);
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    }
}

using OVCompileAndInferRequestTurbo = OVCompileAndInferRequest;

TEST_P(OVCompileAndInferRequestTurbo, CompiledModelTurbo) {
    configuration[intel_npu::turbo.name()] = true;

    auto supportedProperties = core->get_property("NPU", supported_properties.name()).as<std::vector<PropertyName>>();
    bool isTurboSupported =
        std::any_of(supportedProperties.begin(), supportedProperties.end(), [](const PropertyName& property) {
            return property == intel_npu::turbo.name();
        });

    if (isCommandQueueExtSupported()) {
        ASSERT_TRUE(isTurboSupported);
        OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
        auto turbosetting_compiled_model = execNet.get_property(intel_npu::turbo.name());
        OV_ASSERT_NO_THROW(turbosetting_compiled_model = true);
        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        bool is_called = false;
        OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
            ASSERT_EQ(exception_ptr, nullptr);
            is_called = true;
        }));
        OV_ASSERT_NO_THROW(req.start_async());
        OV_ASSERT_NO_THROW(req.wait());
        ASSERT_TRUE(is_called);
    } else {
        OV_EXPECT_THROW_HAS_SUBSTRING(core->compile_model(function, target_device, configuration),
                                      ov::Exception,
                                      "[ NOT_FOUND ] Option 'NPU_TURBO' is not supported for current configuration");
    }
}

using OVCompileAndInferRequestSerializers = OVCompileAndInferRequest;

TEST_P(OVCompileAndInferRequestSerializers, AccurateResults) {
    try {
        execNet = core->compile_model(function, target_device, configuration);
        ov::InferRequest inference_request;
        OV_ASSERT_NO_THROW(inference_request = execNet.create_infer_request());

        const ov::Tensor input = utils::create_tensor(ov::element::f32, ov::Shape{1}, std::vector<float>{1.0f});
        inference_request.set_input_tensor(input);
        OV_ASSERT_NO_THROW(inference_request.infer());

        const ov::Tensor output = inference_request.get_tensor("tensor_output");
        const ov::Tensor expected = utils::create_tensor(ov::element::f32, ov::Shape{1}, std::vector<float>{3.0f});
        OV_ASSERT_NO_THROW(utils::compare(expected, output));
    } catch (const ov::Exception& exception) {
        ASSERT_STR_CONTAINS(
            exception.what(),
            "[ NOT_FOUND ] Option 'NPU_USE_BASE_MODEL_SERIALIZER' is not supported for current configuration");
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
