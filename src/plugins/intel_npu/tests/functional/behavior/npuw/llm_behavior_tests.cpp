// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_engine/simple_llm_pipeline.hpp"
#include "test_engine/mocks/mock_plugins.hpp"
#include "test_engine/mocks/register_in_ov.hpp"
#include "test_engine/models/minicpm4_05b.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/file_util.hpp"

#include <filesystem>

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

namespace {
    const std::vector<int64_t> What_is_OpenVINO =
        {1, 3067, 1410, 6404, 59408, 2097, 59383, 74};
} // anonymous namespace

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

namespace ov {
namespace npuw {
namespace tests {

class LLMBehaviorTestsNPUW : public ::testing::Test {
public:
    void SetUp() override {
        mock_npu_for_prefill_plugin = std::make_shared<MockNpuPluginForPrefill>();
        mock_npu_for_prefill_plugin->create_implementation();
        mock_npu_for_generate_plugin = std::make_shared<MockNpuPluginForGenerate>();
        mock_npu_for_generate_plugin->create_implementation();
        mock_npu_for_lm_head_plugin = std::make_shared<MockNpuPluginForLMHead>();
        mock_npu_for_lm_head_plugin->create_implementation();
        config["++NPUW_LLM_PREFILL_CONFIG"] = ov::AnyMap{{"NPUW_DEVICES", "MockNPUForPrefill"}};
        config["++NPUW_LLM_GENERATE_CONFIG"] = ov::AnyMap{{"NPUW_DEVICES", "MockNPUForGenerate"}};
        config["++NPUW_LLM_SHARED_HEAD_CONFIG"] = ov::AnyMap{{"NPUW_DEVICES", "MockNPUForLMHead"}};
        config["NPUW_LLM_MIN_RESPONSE_LEN"] = 2;
    }

    // Make sure it is called after expectations are set!
    void register_mock_plugins_in_ov() {
        m_shared_objects.push_back(reg_plugin<MockNpuPluginForPrefill>(core, mock_npu_for_prefill_plugin));
        m_shared_objects.push_back(reg_plugin<MockNpuPluginForGenerate>(core, mock_npu_for_generate_plugin));
        m_shared_objects.push_back(reg_plugin<MockNpuPluginForLMHead>(core, mock_npu_for_lm_head_plugin));
    }

protected:
    ov::Core core;
    SimpleLLMPipeline simple_llm;
    std::shared_ptr<MockNpuPluginForPrefill> mock_npu_for_prefill_plugin;
    std::shared_ptr<MockNpuPluginForGenerate> mock_npu_for_generate_plugin;
    std::shared_ptr<MockNpuPluginForLMHead> mock_npu_for_lm_head_plugin;
    ov::AnyMap config;

private:
    std::vector<std::shared_ptr<void>> m_shared_objects;
};
}  // namespace tests
}  // namespace npuw
}  // namespace ov

TEST_F(LLMBehaviorTestsNPUW, LLMBehaviorNPUW_FAST_COMPILE) {
    const std::string model_path = get_minicpm4_05b_path();
    if (model_path == "") {
        GTEST_SKIP() << "Test model is not found, skipping the test!";
    }

    // Set expectations first:
    {
        InSequence seq;
        // Generate model:
        EXPECT_COMPILE_MODEL(mock_npu_for_generate, TIMES(3));
        // Prefill model:
        EXPECT_COMPILE_MODEL(mock_npu_for_prefill, TIMES(3));
        // LM Head model:
        EXPECT_COMPILE_MODEL(mock_npu_for_lm_head, TIMES(1));
    }

    // ------------------------ Prefill model ---------------------------
    // 1 infer request for head:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_prefill, MODEL(0), TIMES(1));  
    // 2 infer requests for function, `create_sync_infer_request()`
    // should be called twice here (2 requests to form pipelining):
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_prefill, MODEL(1), TIMES(2));
    // 1 infer request for tail:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_INFER(mock_npu_for_prefill, MODEL(0), TIMES(1));
    // There are 23 repeated functions, so:
    // Repeated block's 1st infer request is called 12 times:
    EXPECT_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(0), TIMES(12));
    // Repeated block's 2nd infer request (brother of 1st one) is called 11 times:
    EXPECT_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(1), TIMES(11));
    // Tail's infer request is called once:
    EXPECT_INFER(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // ------------------------ Generate model ---------------------------
    // 1 infer request for head:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_generate, MODEL(0), TIMES(1));  
    // `create_sync_infer_request()` should be called 23 times
    // to create 23 separate infer requests here (due to UNFOLD_IREQS),
    // however, we face: "Subgraph[1] requires unpack, unfold can't be done",
    // thus following default JustInferRequest with FUNCALL_ASYNC routine here:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_generate, MODEL(1), TIMES(2));
    // 1 infer request for tail:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_generate, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_INFER(mock_npu_for_generate, MODEL(0), TIMES(1));
    // There are 23 repeated functions, so:
    // Repeated block's 1st infer request is called 12 times:
    EXPECT_INFER_FOR(mock_npu_for_generate, MODEL(1), INFER_REQ(0), TIMES(12));
    // Repeated block's 2nd infer request (brother of 1st one) is called 11 times:
    EXPECT_INFER_FOR(mock_npu_for_generate, MODEL(1), INFER_REQ(1), TIMES(11));
    // Tail's infer request is called once:
    EXPECT_INFER(mock_npu_for_generate, MODEL(2), TIMES(1));

    // ------------------------- LM Head model -----------------------------
    // Or vocabulary projection model.
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_lm_head, MODEL(0), TIMES(1));
    // Called twice: once for prefill and once for generate
    // (as we have to output only 2 tokens in this test).
    EXPECT_INFER(mock_npu_for_lm_head, MODEL(0), TIMES(2));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    simple_llm.initialize(model_path, core, config); 
    EXPECT_NO_THROW(simple_llm.generate(What_is_OpenVINO));
}
 
TEST_F(LLMBehaviorTestsNPUW, LLMBehaviorNPUW_BEST_PERF) {
    const std::string model_path = get_minicpm4_05b_path();
    if (model_path == "") {
        GTEST_SKIP() << "Test model is not found, skipping the test!";
    }

    // Set expectations first:
    {
        InSequence seq;
        // Generate model:
        EXPECT_COMPILE_MODEL(mock_npu_for_generate, TIMES(1));
        // Prefill model:
        EXPECT_COMPILE_MODEL(mock_npu_for_prefill, TIMES(3));
        // LM Head model:
        EXPECT_COMPILE_MODEL(mock_npu_for_lm_head, TIMES(1));
    }

    // ------------------------ Prefill model ---------------------------
    // 1 infer request for head:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_prefill, MODEL(0), TIMES(1));  
    // 2 infer requests for function, `create_sync_infer_request()`
    // should be called twice here (2 requests to form pipelining):
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_prefill, MODEL(1), TIMES(2));
    // 1 infer request for tail:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_INFER(mock_npu_for_prefill, MODEL(0), TIMES(1));
    // There are 23 repeated functions, so:
    // Repeated block's 1st infer request is called 12 times:
    EXPECT_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(0), TIMES(12));
    // Repeated block's 2nd infer request (brother of 1st one) is called 11 times:
    EXPECT_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(1), TIMES(11));
    // Tail's infer request is called once:
    EXPECT_INFER(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // ------------------------ Generate model ---------------------------
    // 1 infer request for the whole generate model:
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_generate, MODEL(0), TIMES(1));  
    // Infer request is called only once:
    EXPECT_INFER(mock_npu_for_generate, MODEL(0), TIMES(1));

    // ------------------------- LM Head model -----------------------------
    // Or vocabulary projection model.
    EXPECT_CREATE_SYNC_INFER_REQ(mock_npu_for_lm_head, MODEL(0), TIMES(1));
    // Called twice: once for prefill and once for generate
    // (as we have to output only 2 tokens in this test).
    EXPECT_INFER(mock_npu_for_lm_head, MODEL(0), TIMES(2));


    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:

    config["NPUW_LLM_GENERATE_HINT"] = "BEST_PERF";
    simple_llm.initialize(model_path, core, config); 
    EXPECT_NO_THROW(simple_llm.generate(What_is_OpenVINO));
}

