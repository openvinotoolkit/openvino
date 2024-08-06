// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_tests.hpp"
#include "openvino/util/shared_object.hpp"

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

#define THROW(...) \
    .WillOnce(Throw(std::runtime_error(__VA_ARGS__)))

#define EXPECT_COMPILE_MODEL(device, times, ...)                        \
    EXPECT_CALL(*device##_plugin,                                       \
        compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)) \
        .Times(times)                                                   \
        __VA_ARGS__                                                     \

#define MODEL(idx) idx

#define EXPECT_CREATE_SYNC_INFER_REQ(device, model_idx, times, ...)                            \
    device##_plugin->set_expectations_to_comp_models(model_idx, [](MockCompiledModel& model) { \
        EXPECT_CALL(model, create_sync_infer_request())                                        \
        .Times(times)                                                                          \
        __VA_ARGS__;                                                                            \
    })

#define EXPECT_INFER(device, model_idx, times, ...) \
    device##_plugin->set_expectations_to_infer_reqs(model_idx, [](MockInferRequest& request) { \
        EXPECT_CALL(request, infer())                                                          \
        .Times(times)                                                                         \
        __VA_ARGS__;                                                                           \
    });

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
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1);
    EXPECT_COMPILE_MODEL(cpu, 0);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUW, CompilationIsFailSafe) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    {
        InSequence s;

        EXPECT_COMPILE_MODEL(npu, 1, THROW("Compilation on MockNPU is failed"));
        EXPECT_COMPILE_MODEL(cpu, 1);
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUW, CompilationIsFailed) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1, THROW("Compilation on MockNPU is failed"));
    EXPECT_COMPILE_MODEL(cpu, 0);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    EXPECT_ANY_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsSuccessful) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1);
    EXPECT_COMPILE_MODEL(cpu, 0);
    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(0), 1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU")); 
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsFailSafe) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    {
        InSequence s;
        EXPECT_COMPILE_MODEL(npu, 1);
        EXPECT_COMPILE_MODEL(cpu, 1);
    }
    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(0), 1, THROW("Infer request creation on MockNPU is failed"));
    EXPECT_CREATE_SYNC_INFER_REQ(cpu, MODEL(0), 1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsFailed) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1);
    EXPECT_COMPILE_MODEL(cpu, 0);

    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(0), 1, THROW("Infer request creation on MockNPU is failed"));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    EXPECT_ANY_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUW, InferIsSuccessful) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1);
    EXPECT_COMPILE_MODEL(cpu, 0);
    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(0), 1);
    EXPECT_INFER(npu, MODEL(0), 1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUW, InferIsFailSafe) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    {
        InSequence seq;
        EXPECT_COMPILE_MODEL(npu, 1);
        EXPECT_COMPILE_MODEL(cpu, 1);
    }
    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(0), 1);
    EXPECT_INFER(npu, MODEL(0), 1, THROW("Infer on MockNPU is failed"));
    EXPECT_CREATE_SYNC_INFER_REQ(cpu, MODEL(0), 1);
    EXPECT_INFER(cpu, MODEL(0), 1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUW, InferIsFailed) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1);
    EXPECT_COMPILE_MODEL(cpu, 0);

    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(0), 1);
    EXPECT_INFER(npu, MODEL(0), 1, THROW("Infer on MockNPU is failed"));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_ANY_THROW(infer_request.infer());
}

using BehaviorTestsNPUWOnlinePartitioning = BehaviorTestsNPUW;
TEST_F(BehaviorTestsNPUWOnlinePartitioning, CompilationIsSuccessful) {
    const auto& model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, Between(2, 12));
    EXPECT_COMPILE_MODEL(cpu, 0);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, CompilationIsFailSafe) {
    model = model_generator.get_model_with_repeated_blocks();
    // Set expectations first:
    {
        InSequence s;

        EXPECT_COMPILE_MODEL(npu, 1, THROW("Compilation on MockNPU is failed"));
        EXPECT_COMPILE_MODEL(cpu, 1);
        EXPECT_COMPILE_MODEL(npu, Between(2, 11));
        
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, CompilationIsFailed) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 1, THROW("Compilation on MockNPU is failed"));
    EXPECT_COMPILE_MODEL(cpu, 0);    

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    EXPECT_ANY_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferRequestCreationIsSuccessful) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 12);
    EXPECT_COMPILE_MODEL(cpu, 0);

    for (int i = 0;  i < 12; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(i), 1);
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    compiled_model.create_infer_request();
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferRequestCreationIsFailSafe) {
    model = model_generator.get_model_with_repeated_blocks();
    // Set expectations first:
    {
        InSequence s;

        EXPECT_COMPILE_MODEL(npu, 12);
        EXPECT_COMPILE_MODEL(cpu, 1);
    }

    for (int i = 0;  i < 11; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(i), 1);
    }
    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(11), 1, THROW("Infer request creation on MockNPU is failed"));
    EXPECT_CREATE_SYNC_INFER_REQ(cpu, MODEL(0), 1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferRequestCreationIsFailed) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 12);
    EXPECT_COMPILE_MODEL(cpu, 0);

    for (int i = 0;  i < 11; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(i), 1);
    }
    EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(11), 1, THROW("Infer request creation on MockNPU is failed"));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    EXPECT_ANY_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferIsSuccessful) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(npu, 12);
    EXPECT_COMPILE_MODEL(cpu, 0);

    for (int i = 0;  i < 12; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(i), 1);
        EXPECT_INFER(npu, MODEL(i), 1);
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferIsFailSafe) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    {
        InSequence seq;
        EXPECT_COMPILE_MODEL(npu, 12);
        EXPECT_COMPILE_MODEL(cpu, 1);
    }
    for (int i = 0;  i < 12; i++) {
       EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(i), 1);
    }
    for (int i = 0;  i < 11; i++) {
        EXPECT_INFER(npu, MODEL(i), 1);
    }
    EXPECT_INFER(npu, MODEL(11), 1, THROW("Infer on MockNPU is failed"));

    EXPECT_CREATE_SYNC_INFER_REQ(cpu, MODEL(0), 1);
    EXPECT_INFER(cpu, MODEL(0), 1);

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferIsFailed) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    {
        InSequence seq;
        EXPECT_COMPILE_MODEL(npu, 12);
        EXPECT_COMPILE_MODEL(cpu, 0);
    }
    for (int i = 0;  i < 12; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(npu, MODEL(i), 1);
    }
    for (int i = 0;  i < 11; i++) {
        EXPECT_INFER(npu, MODEL(i), 1);
    }
    EXPECT_INFER(npu, MODEL(11), 1, THROW("Infer on MockNPU is failed"));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_ANY_THROW(infer_request.infer());
}
