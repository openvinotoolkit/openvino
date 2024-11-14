// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_tests.hpp"
#include "comparators/nrmse.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/file_util.hpp"

#include <filesystem>

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

#define TIMES(times) times 

#define THROW(...) \
    .WillOnce(Throw(std::runtime_error(__VA_ARGS__)))

#define EXPECT_COMPILE_MODEL(device, times, ...)                        \
    EXPECT_CALL(*device##_plugin,                                       \
        compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)) \
        .Times(times)                                                   \
        __VA_ARGS__                                                     \

#define MODEL(idx) idx

#define INFER_REQ(idx) idx

#define EXPECT_CREATE_SYNC_INFER_REQ(device, model_idx, times, ...)                            \
    device##_plugin->set_expectations_to_comp_models(model_idx, [](MockCompiledModel& model) { \
        EXPECT_CALL(model, create_sync_infer_request())                                        \
        .Times(times)                                                                          \
        __VA_ARGS__;                                                                           \
    })

#define EXPECT_INFER(device, model_idx, times, ...) \
    device##_plugin->set_expectations_to_infer_reqs(model_idx, 0, [](MockInferRequest& request) { \
        EXPECT_CALL(request, infer())                                                             \
        .Times(times)                                                                             \
        __VA_ARGS__;                                                                              \
    });

#define EXPECT_INFER_FOR(device, model_idx, req_idx, times, ...) \
    device##_plugin->set_expectations_to_infer_reqs(model_idx, req_idx, [](MockInferRequest& request) { \
        EXPECT_CALL(request, infer())                                                                   \
        .Times(times)                                                                                   \
        __VA_ARGS__;                                                                                    \
    });

TEST_F(BehaviorTestsNPUW, TestInfrastructureIsCorrect) {
    // Set expectations first:
    EXPECT_CALL(*mock_npu_plugin, get_property).Times(AnyNumber());
    EXPECT_CALL(*mock_npu_plugin, get_property(std::string("AVAILABLE_DEVICES"), _)).Times(1);
    EXPECT_CALL(*mock_cpu_plugin, get_property).Times(AnyNumber());
    EXPECT_CALL(*mock_cpu_plugin, get_property(std::string("AVAILABLE_DEVICES"), _)).Times(1);

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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

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

        EXPECT_COMPILE_MODEL(mock_npu, TIMES(1), THROW("Compilation on MockNPU is failed"));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(1));
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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1), THROW("Compilation on MockNPU is failed"));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    EXPECT_ANY_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUW, InferRequestCreationIsSuccessful) {
    model = model_generator.get_model_with_one_op();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1));

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
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(1));
    }
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1), THROW("Infer request creation on MockNPU is failed"));
    EXPECT_CREATE_SYNC_INFER_REQ(mock_cpu, MODEL(0), TIMES(1));

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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1), THROW("Infer request creation on MockNPU is failed"));

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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1));
    EXPECT_INFER(mock_npu, MODEL(0), TIMES(1));

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
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(1));
    }
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1));
    EXPECT_INFER(mock_npu, MODEL(0), TIMES(1), THROW("Infer on MockNPU is failed"));
    EXPECT_CREATE_SYNC_INFER_REQ(mock_cpu, MODEL(0), TIMES(1));
    EXPECT_INFER(mock_cpu, MODEL(0), TIMES(1));

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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1));
    EXPECT_INFER(mock_npu, MODEL(0), TIMES(1), THROW("Infer on MockNPU is failed"));

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
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));    
    use_npuw_props.emplace(partitioning::online::min_size(12));
    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, CompilationIsFailSafe) {
    model = model_generator.get_model_with_repeated_blocks();
    // Set expectations first:
    {
        InSequence s;

        EXPECT_COMPILE_MODEL(mock_npu, TIMES(1), THROW("Compilation on MockNPU is failed"));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(1));
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(11));
        
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU,MockCPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, CompilationIsFailed) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1), THROW("Compilation on MockNPU is failed"));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));    

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    EXPECT_ANY_THROW(core.compile_model(model, "NPU", use_npuw_props));
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferRequestCreationIsSuccessful) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    for (int i = 0;  i < 12; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, InferRequestCreationIsFailSafe) {
    model = model_generator.get_model_with_repeated_blocks();
    // Set expectations first:
    {
        InSequence s;

        EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(1));
    }

    for (int i = 0;  i < 11; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
    }
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(11), TIMES(1), THROW("Infer request creation on MockNPU is failed"));
    EXPECT_CREATE_SYNC_INFER_REQ(mock_cpu, MODEL(0), TIMES(1));

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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    for (int i = 0;  i < 11; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
    }
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(11), TIMES(1), THROW("Infer request creation on MockNPU is failed"));

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
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    for (int i = 0;  i < 12; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
        EXPECT_INFER(mock_npu, MODEL(i), TIMES(1));
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
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(1));
    }
    for (int i = 0;  i < 12; i++) {
       EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
    }
    for (int i = 0;  i < 11; i++) {
        EXPECT_INFER(mock_npu, MODEL(i), TIMES(1));
    }
    EXPECT_INFER(mock_npu, MODEL(11), TIMES(1), THROW("Infer on MockNPU is failed"));

    EXPECT_CREATE_SYNC_INFER_REQ(mock_cpu, MODEL(0), TIMES(1));
    EXPECT_INFER(mock_cpu, MODEL(0), TIMES(1));

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
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(12));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));
    }
    for (int i = 0;  i < 12; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
    }
    for (int i = 0;  i < 11; i++) {
        EXPECT_INFER(mock_npu, MODEL(i), TIMES(1));
    }
    EXPECT_INFER(mock_npu, MODEL(11), TIMES(1), THROW("Infer on MockNPU is failed"));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_ANY_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, RepeatedBlocksAreFolded) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    {
        InSequence seq;
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(3));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));
    }
    for (int i = 0;  i < 3; i++) {
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i), TIMES(1));
    }

    // 1st model infer request is called once -- head
    EXPECT_INFER(mock_npu, MODEL(0), TIMES(1));
    // 2nd model infer request is called 10 times -- repeated block
    EXPECT_INFER(mock_npu, MODEL(1), TIMES(10));
    // 3rd model infer request is called once -- tail
    EXPECT_INFER(mock_npu, MODEL(2), TIMES(1));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    use_npuw_props.emplace(partitioning::fold(true));
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

TEST_F(BehaviorTestsNPUWOnlinePartitioning, FoldingAndPipelining) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    {
        InSequence seq;
        EXPECT_COMPILE_MODEL(mock_npu, TIMES(3));
        EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));
    }

    // 1 infer request for head:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(0), TIMES(1));  
    // 2 infer requests for function, `create_sync_infer_request()`
    // should be called twice here:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(1), TIMES(2));
    // 1 infer request for tail:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_INFER(mock_npu, MODEL(0), TIMES(1));

    // Repeated block's model 1st infer request is called 5 times:
    EXPECT_INFER_FOR(mock_npu, MODEL(1), INFER_REQ(0), TIMES(5));
    // Repeated block's model 2nd infer request (brother of 1st one) is called 5 times:
    EXPECT_INFER_FOR(mock_npu, MODEL(1), INFER_REQ(1), TIMES(5));

    // Tail's infer request is called once:
    EXPECT_INFER(mock_npu, MODEL(2), TIMES(1));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::min_size(12));
    use_npuw_props.emplace(partitioning::fold(true));
    use_npuw_props.emplace(partitioning::dcoff_type("f16"));
    use_npuw_props.emplace(partitioning::dcoff_with_scale(true));  
    use_npuw_props.emplace(funcall_async(true));  
    auto compiled_model = core.compile_model(model, "NPU", use_npuw_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
}

using BehaviorTestsNPUWOfflinePartitioning = BehaviorTestsNPUW;
TEST_F(BehaviorTestsNPUWOfflinePartitioning, CompilationIsSuccessful) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    // For plan generation and execution (twice compiled):
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(24));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Generate plan:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::dump_plan("test_partitioning.xml"));

    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
    EXPECT_TRUE(std::filesystem::exists("test_partitioning.xml"));

    // Do the actual test:
    ov::AnyMap offline_plan_props = { ::ov::intel_npu::use_npuw(true),
                                      devices("MockNPU"),
                                      partitioning::plan("test_partitioning.xml") };
    ov::CompiledModel compiled_model;
    EXPECT_NO_THROW(compiled_model = core.compile_model(model, "NPU", offline_plan_props));
    EXPECT_EQ("test_partitioning.xml",
        compiled_model.get_property(partitioning::plan.name()).as<std::string>());
    EXPECT_TRUE(std::filesystem::remove("test_partitioning.xml"));
}

TEST_F(BehaviorTestsNPUWOfflinePartitioning, InferRequestCreationIsSuccessful) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    // For plan generation and execution (twice compiled):
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(24));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    for (int i = 0;  i < 12; i++) {
        // First 12 models are compiled only to generate partitioning plan:
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i + 12), 1);
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Generate plan:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::dump_plan("test_partitioning.xml"));
    use_npuw_props.emplace(partitioning::online::pipeline("REP"));

    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
    EXPECT_TRUE(std::filesystem::exists("test_partitioning.xml"));

    // Do the actual test:
    ov::AnyMap offline_plan_props = { ::ov::intel_npu::use_npuw(true),
                                      devices("MockNPU"),
                                      partitioning::plan("test_partitioning.xml") };
    auto compiled_model = core.compile_model(model, "NPU", offline_plan_props);
    EXPECT_NO_THROW(compiled_model.create_infer_request());
    EXPECT_EQ("test_partitioning.xml",
        compiled_model.get_property(partitioning::plan.name()).as<std::string>());
    EXPECT_TRUE(std::filesystem::remove("test_partitioning.xml"));
}

TEST_F(BehaviorTestsNPUWOfflinePartitioning, InferIsSuccessful) {
    model = model_generator.get_model_with_repeated_blocks();

    // Set expectations first:
    // For plan generation and execution (twice compiled):
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(24));
    EXPECT_COMPILE_MODEL(mock_cpu, TIMES(0));

    for (int i = 0;  i < 12; i++) {
        // First 12 models are compiled only to generate partitioning plan:
        EXPECT_CREATE_SYNC_INFER_REQ(mock_npu, MODEL(i + 12), 1);
        EXPECT_INFER(mock_npu, MODEL(i + 12), 1);
    }

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Generate plan:
    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::online::dump_plan("test_partitioning.xml"));
    use_npuw_props.emplace(partitioning::online::pipeline("REP"));

    EXPECT_NO_THROW(core.compile_model(model, "NPU", use_npuw_props));
    EXPECT_TRUE(std::filesystem::exists("test_partitioning.xml"));

    // Do the actual test:
    ov::AnyMap offline_plan_props = { ::ov::intel_npu::use_npuw(true),
                                      devices("MockNPU"),
                                      partitioning::plan("test_partitioning.xml") };
    auto compiled_model = core.compile_model(model, "NPU", offline_plan_props);
    auto infer_request = compiled_model.create_infer_request();
    EXPECT_NO_THROW(infer_request.infer());
    EXPECT_EQ("test_partitioning.xml",
        compiled_model.get_property(partitioning::plan.name()).as<std::string>());
    EXPECT_TRUE(std::filesystem::remove("test_partitioning.xml"));  
}

TEST(BehaviorTestsNPUWAccuracy, RepAndNonePartPipesGiveSameResults) {
    // Register TEMPLATE plugin in OpenVINO:
    ov::Core core;
    auto plugin_path =
        ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                           std::string(ov::test::utils::TEMPLATE_LIB)
                                           + OV_BUILD_POSTFIX);
    if (!ov::util::file_exists(plugin_path)) {
        OPENVINO_THROW("Plugin: " + plugin_path + " does not exists!");
    }
    core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);

    // Create model:
    ModelGenerator model_generator;
    const auto &model = model_generator.get_model_with_repeated_blocks(32);

    // Do the test:
    ov::AnyMap online_rep_props { ::ov::intel_npu::use_npuw(true),
                                  devices("TEMPLATE"),
                                  partitioning::online::pipeline("REP") };
    ov::AnyMap online_none_props { ::ov::intel_npu::use_npuw(true),
                                   devices("TEMPLATE"),
                                   partitioning::online::pipeline("NONE") };

    auto rep_compiled_model = core.compile_model(model, "NPU", online_rep_props);
    auto none_compiled_model = core.compile_model(model, "NPU", online_none_props);
    auto rep_infer_request = rep_compiled_model.create_infer_request();
    auto none_infer_request = rep_compiled_model.create_infer_request();

    set_random_inputs<int32_t>(rep_infer_request);
    set_random_inputs<int32_t>(none_infer_request);

    rep_infer_request.infer();
    none_infer_request.infer();

    metrics::NRMSE nrmse(0.01);

    for (const auto& output : rep_infer_request.get_compiled_model().outputs()) {
        const auto& rep_tensor = rep_infer_request.get_tensor(output);
        const auto& none_tensor = none_infer_request.get_tensor(output);
        EXPECT_TRUE(nrmse(rep_tensor, none_tensor));
    }
}

TEST_F(BehaviorTestsNPUW, CanSayNoToPMMProperty) {
    // Create model:
    model = model_generator.get_model_with_one_op();

    // Set expectation to npu plugin:
    EXPECT_COMPILE_MODEL(mock_npu, TIMES(1));

    // Register mock npu plugin in OpenVINO:
    register_mock_plugins_in_ov();

    use_npuw_props.emplace(devices("MockNPU"));
    use_npuw_props.emplace(partitioning::par_matmul_merge_dims("NO"));

    ov::CompiledModel compiled_model;
    EXPECT_NO_THROW(compiled_model = core.compile_model(model, "NPU", use_npuw_props));
    auto prop = compiled_model.get_property(partitioning::par_matmul_merge_dims.name());
    EXPECT_EQ("NO", prop.as<std::string>());
}
