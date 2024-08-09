// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_tests.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "npuw_private_properties.hpp"

using namespace testing;
using namespace ov::npuw::tests;

std::shared_ptr<ov::Model> BehaviorTestsNPUW::create_example_model() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset11::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

TEST_F(BehaviorTestsNPUW, TestInfrastructureIsCorrect) {
    // Set expectations first:
    EXPECT_CALL(*npu_plugin, get_property).Times(AnyNumber());
    EXPECT_CALL(*npu_plugin, get_property(std::string("AVAILABLE_DEVICES"), _)).Times(1);
    EXPECT_CALL(*cpu_plugin, get_property).Times(AnyNumber());
    EXPECT_CALL(*cpu_plugin, get_property(std::string("AVAILABLE_DEVICES"), _)).Times(1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    std::vector<std::string> mock_reference_dev = {{"MockNPU.0"}, {"MockNPU.1"}, {"MockNPU.2"},
                                                   {"MockCPU.0"}, {"MockCPU.1"}};
    auto available_devices = core.get_available_devices();
    for (auto device : available_devices) {
        auto it = std::find(mock_reference_dev.begin(), mock_reference_dev.end(), device);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }

    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(BehaviorTestsNPUW, CompilationIsSuccessful) {
    // Set expectations first:
    EXPECT_CALL(*npu_plugin,
        compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*cpu_plugin,
        compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU")};
    EXPECT_NO_THROW(core.compile_model(model, "NPU", npuw_properties));
}

TEST_F(BehaviorTestsNPUW, CompilationIsFailSafe) {
    // Set expectations first:
    {
        InSequence s;

        EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
                .Times(1)
                .WillOnce(Throw(std::runtime_error("Compilation on MockNPU is failed")));
        EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU,MockCPU")};
    EXPECT_NO_THROW(core.compile_model(model, "NPU", npuw_properties));
}

TEST_F(BehaviorTestsNPUW, CompilationIsFailed) {
    // Set expectations first:
    EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _))
            .Times(1)
            .WillOnce(Throw(std::runtime_error("Compilation on MockNPU is failed")));
    EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU")};
    EXPECT_ANY_THROW(core.compile_model(model, "NPU", npuw_properties));
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsSuccessful) {
    // Set expectations first:
    EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
    npu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request()).Times(1);
    });

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU")};
    auto compiled_model = core.compile_model(model, "NPU", npuw_properties);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsFailSafe) {
    // Set expectations first:
    {
        InSequence s;
        EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    }
    npu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request())
            .Times(1)
            .WillOnce(Throw(std::runtime_error("Infer request creation on MockNPU is failed")));
    });
    cpu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request()).Times(1);
    });

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU,MockCPU")};
    auto compiled_model = core.compile_model(model, "NPU", npuw_properties);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsFailed) {
    // Set expectations first:
    EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);

    npu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request())
                .Times(1)
                .WillOnce(Throw(std::runtime_error("Infer request creation on MockNPU is failed")));
    });

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU")};
    auto compiled_model = core.compile_model(model, "NPU", npuw_properties);
    EXPECT_ANY_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUW, InferIsSuccessful) {
    // Set expectations first:
    EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
    npu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request()).Times(1);
    });
    npu_plugin->set_expectations_to_infer_reqs(0, [](MockInferRequest& request) {
        EXPECT_CALL(request, infer()).Times(1);
    });

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU")};
    auto compiled_model = core.compile_model(model, "NPU", npuw_properties);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUW, InferIsFailSafe) {
    // Set expectations first:
    {
        InSequence seq;
        EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
    }
    npu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request()).Times(1);
    });
    npu_plugin->set_expectations_to_infer_reqs(0, [](MockInferRequest& request) {
        EXPECT_CALL(request, infer())
            .Times(1)
            .WillOnce(Throw(std::runtime_error("Infer on MockNPU is failed")));
    });   
    cpu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request()).Times(1);
    });
    cpu_plugin->set_expectations_to_infer_reqs(0, [](MockInferRequest& request) {
        EXPECT_CALL(request, infer()).Times(1);
    });

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU,MockCPU")};
    auto compiled_model = core.compile_model(model, "NPU", npuw_properties);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUW, InferIsFailed) {
    // Set expectations first:
    {
        InSequence seq;
        EXPECT_CALL(*npu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(1);
        EXPECT_CALL(*cpu_plugin, compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)).Times(0);
    }
    npu_plugin->set_expectations_to_comp_models(0, [](MockCompiledModel& model) {
        EXPECT_CALL(model, create_sync_infer_request()).Times(1);
    });
    npu_plugin->set_expectations_to_infer_reqs(0, [](MockInferRequest& request) {
        EXPECT_CALL(request, infer())
            .Times(1)
            .WillOnce(Throw(std::runtime_error("Infer on MockNPU is failed")));
    });

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    auto npuw_properties = 
        ov::AnyMap{ov::intel_npu::use_npuw(true),
                   ov::intel_npu::npuw::devices("MockNPU")};
    auto compiled_model = core.compile_model(model, "NPU", npuw_properties);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_ANY_THROW(infer_request.infer());
}
