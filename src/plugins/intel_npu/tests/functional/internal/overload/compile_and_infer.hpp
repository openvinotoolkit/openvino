// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <common_test_utils/test_assertions.hpp>
#include <sstream>

#include "base/ov_behavior_test_utils.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/except.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/properties.hpp"

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
    static std::string getTestCaseName(testing::TestParamInfo<CompileAndInferRequestParams> obj) {
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
        // Skip test according to plugin specific disabledTestPatterns() (if any)
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
        OV_EXPECT_THROW_HAS_SUBSTRING(core->compile_model(function, target_device, configuration),
                                      ov::Exception,
                                      "WorkloadType property is not supported by the current Driver Version!");
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
                                      "WorkloadType property is not supported by the current Driver Version!");
    }
}

TEST_P(OVCompileAndInferRequest, CompiledModelWorkloadTypeDelayedExecutor) {
    configuration[intel_npu::defer_weights_load.name()] = true;
    OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
    ov::AnyMap modelConfiguration;
    modelConfiguration[workload_type.name()] = WorkloadType::DEFAULT;
    OV_ASSERT_NO_THROW(execNet.set_property(modelConfiguration));

    if (isCommandQueueExtSupported()) {
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
        OV_EXPECT_THROW_HAS_SUBSTRING(execNet.create_infer_request(),
                                      ov::Exception,
                                      "WorkloadType property is not supported by the current Driver Version!");
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
        auto cr_ex = configuration.find(intel_npu::defer_weights_load.name());
        if (cr_ex->second.as<bool>() == false) {
            OV_EXPECT_THROW_HAS_SUBSTRING(core->compile_model(function, target_device, configuration),
                                          ov::Exception,
                                          "Turbo is not supported by the current driver");
        } else {
            OV_ASSERT_NO_THROW(execNet = core->compile_model(function, target_device, configuration));
            OV_EXPECT_THROW_HAS_SUBSTRING(execNet.create_infer_request(),
                                          ov::Exception,
                                          "Turbo is not supported by the current driver");
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
