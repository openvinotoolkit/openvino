// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_engine/simple_llm_pipeline.hpp"
#include "test_engine/mocks/mock_factory.hpp"
#include "test_engine/mocks/mock_plugins.hpp"
#include "test_engine/mocks/register_in_ov.hpp"
#include "test_engine/models/model_builder.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/file_util.hpp"

#include <filesystem>

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

#define MODEL(idx) idx

#define INFER_REQ(idx) idx

#define TIMES(times) times 

#define THROW(...) \
    .WillOnce(Throw(std::runtime_error(__VA_ARGS__)))

#define EXPECT_FACTORY_CREATE_MODEL(times, ...)                         \
    EXPECT_CALL(*mock_npuw_factory, create(_, _, _))                    \
        .Times(times)                                                   \
        __VA_ARGS__                                                     \

#define EXPECT_FACTORY_MODEL_CREATE_INFER_REQ(model_idx, times, ...)                                 \
    mock_npuw_factory->set_expectations_to_comp_models(model_idx, [](MockNpuwCompiledModel& model) { \
        EXPECT_CALL(model, create_sync_infer_request())                                              \
        .Times(times)                                                                                \
        __VA_ARGS__;                                                                                 \
    })

#define EXPECT_FACTORY_MODEL_INFER(model_idx, times, ...)                                               \
    mock_npuw_factory->set_expectations_to_infer_reqs(model_idx, 0, [](MockNpuwInferRequest& request) { \
        EXPECT_CALL(request, infer())                                                                   \
        .Times(times)                                                                                   \
        __VA_ARGS__;                                                                                    \
    });


#define EXPECT_PLUGIN_COMPILE_MODEL(device, times, ...)                 \
    EXPECT_CALL(*device##_plugin,                                       \
        compile_model(A<const std::shared_ptr<const ov::Model>&>(), _)) \
        .Times(times)                                                   \
        __VA_ARGS__                                                     \

#define EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(device, model_idx, times, ...)                    \
    device##_plugin->set_expectations_to_comp_models(model_idx, [](MockCompiledModel& model) { \
        EXPECT_CALL(model, create_sync_infer_request())                                        \
        .Times(times)                                                                          \
        __VA_ARGS__;                                                                           \
    })

#define EXPECT_PLUGIN_MODEL_INFER(device, model_idx, times, ...)                                  \
    device##_plugin->set_expectations_to_infer_reqs(model_idx, 0, [](MockInferRequest& request) { \
        EXPECT_CALL(request, infer())                                                             \
        .Times(times)                                                                             \
        __VA_ARGS__;                                                                              \
    });

#define EXPECT_PLUGIN_MODEL_INFER_FOR(device, model_idx, req_idx, times, ...)                            \
    device##_plugin->set_expectations_to_infer_reqs(model_idx, req_idx, [](MockInferRequest& request) { \
        EXPECT_CALL(request, infer())                                                                   \
        .Times(times)                                                                                   \
        __VA_ARGS__;                                                                                    \
    });

namespace {
    const std::vector<int64_t> What_is_OpenVINO =
        {1, 3067, 1410, 6404, 59408, 2097, 59383, 74};
} // anonymous namespace


namespace ov {
namespace npuw {
namespace tests {
class LLMBehaviorTestsNPUW : public ::testing::Test {
public:
    void SetUp() override {
        ov::test::npuw::ModelBuilder model_builder;
        ov::test::npuw::ModelConfig model_config;
        model_config.num_layers = 12;
        ov_model = model_builder.build_model(model_config);

        mock_npuw_factory = std::make_shared<MockNpuwCompiledModelFactory>();
        mock_npuw_factory->create_implementation();
        config["NPUW_DEVICES"] = ov::AnyMap{{"NPUW_DEVICES", "CPU"}};
        config["NPUW_LLM_MIN_RESPONSE_LEN"] = 2;
        config["NPUW_LLM_NPUWMODEL_FACTORY_PTR"] = mock_npuw_factory;
    }

protected:
    std::shared_ptr<ov::Model> ov_model;
    ov::Core core;
    SimpleLLMPipeline simple_llm;
    std::shared_ptr<MockNpuwCompiledModelFactory> mock_npuw_factory;
    ov::AnyMap config;

private:
    std::vector<std::shared_ptr<void>> m_shared_objects;
};
}  // namespace tests
}  // namespace npuw
}  // namespace ov

TEST_F(LLMBehaviorTestsNPUW, LLMBehaviorNPUW) {
    // Set expectations first:
    EXPECT_FACTORY_CREATE_MODEL(TIMES(3));

    // 1. Infer request for prefill model:
    EXPECT_FACTORY_MODEL_CREATE_INFER_REQ(MODEL(0), TIMES(1));
    // 2. Infer request for generate model:
    EXPECT_FACTORY_MODEL_CREATE_INFER_REQ(MODEL(1), TIMES(1));
    // 3. Infer request for LM head model:
    EXPECT_FACTORY_MODEL_CREATE_INFER_REQ(MODEL(2), TIMES(1));

    // 1. Infer of prefill model:
    EXPECT_FACTORY_MODEL_INFER(MODEL(0), TIMES(1));
    // 2. Infer of generate model:
    EXPECT_FACTORY_MODEL_INFER(MODEL(1), TIMES(1));
    // 3. Infer of LM head model:
    EXPECT_FACTORY_MODEL_INFER(MODEL(2), TIMES(1));

    // Do the actual test:
    simple_llm.initialize(ov_model, core, config); 
    EXPECT_NO_THROW(simple_llm.generate(What_is_OpenVINO));

    // Make non-GMock related checks:
    EXPECT_EQ(3u, mock_npuw_factory->m_ov_models.size());
    std::shared_ptr<ov::Model> ov_prefill_model = mock_npuw_factory->m_ov_models[0];
    auto prefill_input_ids_shape = ov_prefill_model->inputs()[0].get_shape();
    auto prefill_attention_mask_shape = ov_prefill_model->inputs()[1].get_shape();
    EXPECT_EQ(ov::Shape(1, 128), prefill_input_ids_shape);
    EXPECT_EQ(ov::Shape(1, 128), prefill_attention_mask_shape);
    {
        bool npuw_output_embed_found {false};
        for (auto output : ov_prefill_model->outputs()) {
            auto names = output.get_names();
            if (names.count("npuw_output_embed")) {
                npuw_output_embed_found = true;
                break;
            }
        }
        EXPECT_TRUE(npuw_output_embed_found);
    }

    std::shared_ptr<ov::Model> ov_generate_model = mock_npuw_factory->m_ov_models[1];
    auto generate_input_ids_shape = ov_generate_model->inputs()[0].get_shape();
    auto generate_attention_mask_shape = ov_generate_model->inputs()[1].get_shape();
    EXPECT_EQ(ov::Shape(1, 1), generate_input_ids_shape);
    EXPECT_EQ(ov::Shape(1, 132), generate_attention_mask_shape);
    {
        bool npuw_output_embed_found {false};
        for (auto output : ov_generate_model->outputs()) {
            auto names = output.get_names();
            if (names.count("npuw_output_embed")) {
                npuw_output_embed_found = true;
                break;
            }
        }
        EXPECT_TRUE(npuw_output_embed_found);
    }
}


namespace ov {
namespace npuw {
namespace tests {
class LLMLowLevelBehaviorTestsNPUW : public ::testing::Test {
public:
    void SetUp() override {
        ov::test::npuw::ModelBuilder model_builder;
        ov::test::npuw::ModelConfig model_config;
        model_config.num_layers = 12;
        ov_model = model_builder.build_model(model_config);
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
    std::shared_ptr<ov::Model> ov_model;
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

TEST_F(LLMLowLevelBehaviorTestsNPUW, LLMLowLevelBehaviorNPUW_FAST_COMPILE) {
    // Set expectations first:
    {
        InSequence seq;
        // Generate model:
        EXPECT_PLUGIN_COMPILE_MODEL(mock_npu_for_generate, TIMES(3));
        // Prefill model:
        EXPECT_PLUGIN_COMPILE_MODEL(mock_npu_for_prefill, TIMES(3));
        // LM Head model:
        EXPECT_PLUGIN_COMPILE_MODEL(mock_npu_for_lm_head, TIMES(1));
    }

    // ------------------------ Prefill model ---------------------------
    // 1 infer request for head:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_prefill, MODEL(0), TIMES(1));  
    // 2 infer requests for function, `create_sync_infer_request()`
    // should be called twice here (2 requests to form pipelining):
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_prefill, MODEL(1), TIMES(2));
    // 1 infer request for tail:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_prefill, MODEL(0), TIMES(1));
    // There are 11 repeated functions, so:
    // Repeated block's 1st infer request is called 6 times:
    EXPECT_PLUGIN_MODEL_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(0), TIMES(6));
    // Repeated block's 2nd infer request (brother of 1st one) is called 5 times:
    EXPECT_PLUGIN_MODEL_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(1), TIMES(5));
    // Tail's infer request is called once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // ------------------------ Generate model ---------------------------
    // 1 infer request for head:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_generate, MODEL(0), TIMES(1));  
    // `create_sync_infer_request()` should be called 11 times
    // to create 11 separate infer requests here (due to UNFOLD_IREQS):
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_generate, MODEL(1), TIMES(11));
    // 1 infer request for tail:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_generate, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_generate, MODEL(0), TIMES(1));
    // Different 11 infer requests of 2nd submodel should be called:
    for (int i = 0; i < 11; ++i) {
        EXPECT_PLUGIN_MODEL_INFER_FOR(mock_npu_for_generate, MODEL(1), INFER_REQ(i), TIMES(1));
    }
    // Tail's infer request is called once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_generate, MODEL(2), TIMES(1));

    // ------------------------- LM Head model -----------------------------
    // Or vocabulary projection model.
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_lm_head, MODEL(0), TIMES(1));
    // Called twice: once for prefill and once for generate
    // (as we have to output only 2 tokens in this test).
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_lm_head, MODEL(0), TIMES(2));

    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:
    simple_llm.initialize(ov_model, core, config); 
    EXPECT_NO_THROW(simple_llm.generate(What_is_OpenVINO));
}
 
TEST_F(LLMLowLevelBehaviorTestsNPUW, LLMLowLevelBehaviorNPUW_BEST_PERF) {
    // Set expectations first:
    {
        InSequence seq;
        // Generate model:
        EXPECT_PLUGIN_COMPILE_MODEL(mock_npu_for_generate, TIMES(1));
        // Prefill model:
        EXPECT_PLUGIN_COMPILE_MODEL(mock_npu_for_prefill, TIMES(3));
        // LM Head model:
        EXPECT_PLUGIN_COMPILE_MODEL(mock_npu_for_lm_head, TIMES(1));
    }

    // ------------------------ Prefill model ---------------------------
    // 1 infer request for head:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_prefill, MODEL(0), TIMES(1));  
    // 2 infer requests for function, `create_sync_infer_request()`
    // should be called twice here (2 requests to form pipelining):
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_prefill, MODEL(1), TIMES(2));
    // 1 infer request for tail:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // Head's infer request is called once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_prefill, MODEL(0), TIMES(1));
    // There are 11 repeated functions, so:
    // Repeated block's 1st infer request is called 6 times:
    EXPECT_PLUGIN_MODEL_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(0), TIMES(6));
    // Repeated block's 2nd infer request (brother of 1st one) is called 5 times:
    EXPECT_PLUGIN_MODEL_INFER_FOR(mock_npu_for_prefill, MODEL(1), INFER_REQ(1), TIMES(5));
    // Tail's infer request is called once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_prefill, MODEL(2), TIMES(1));

    // ------------------------ Generate model ---------------------------
    // 1 infer request for the whole generate model:
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_generate, MODEL(0), TIMES(1));  
    // Infer request is called only once:
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_generate, MODEL(0), TIMES(1));

    // ------------------------- LM Head model -----------------------------
    // Or vocabulary projection model.
    EXPECT_PLUGIN_MODEL_CREATE_INFER_REQ(mock_npu_for_lm_head, MODEL(0), TIMES(1));
    // Called twice: once for prefill and once for generate
    // (as we have to output only 2 tokens in this test).
    EXPECT_PLUGIN_MODEL_INFER(mock_npu_for_lm_head, MODEL(0), TIMES(2));


    // Register mock objects as plugins in OpenVINO:
    register_mock_plugins_in_ov();

    // Do the actual test:

    config["NPUW_LLM_GENERATE_HINT"] = "BEST_PERF";
    simple_llm.initialize(ov_model, core, config); 
    EXPECT_NO_THROW(simple_llm.generate(What_is_OpenVINO));
}
