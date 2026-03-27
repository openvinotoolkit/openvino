// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <numeric>

#include "functional_test_utils/skip_tests_config.hpp"
#include "npuw/test_engine/models/model_builder.hpp"
#include "npuw/test_engine/simple_llm_pipeline.hpp"

class MultiOutsLLMSmokeBehaviorNPUW : public ::testing::Test {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        // Test will only work if CPU plugin is enabled.
        #if !defined(OPENVINO_STATIC_LIBRARY) && defined(WITH_CPU_PLUGIN) && \
            (defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64))
                const char* cpu_plugin_file_name = "openvino_intel_cpu_plugin";
                // Register CPU plugin in OpenVINO:
                try {
                    core.register_plugin(std::string(cpu_plugin_file_name) + OV_BUILD_POSTFIX, "CPU");
                } catch (ov::Exception& ex) {
                    if (std::string{ex.what()}.find("Device with \"CPU\"  is already registered in the OpenVINO Runtime")
                        == std::string::npos) {
                        throw ex;
                    }
                }
        #else
            GTEST_SKIP() << "CPU plugin is not enabled or platform requirements are not met, skipping the test!";
        #endif // not OPENVINO_STATIC_LIBRARY && WITH_CPU_PLUGIN &&
            // (defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64));

        chunked_prefill_props = 
            {{"NPU_USE_NPUW", "YES"},
                {"NPUW_LLM", "YES"},
                {"NPUW_DEVICES", "CPU"},
                {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}};


        input_ids.resize(128);
        std::iota(input_ids.begin(), input_ids.end(), 1);
    }

    const std::shared_ptr<ov::Model> build_multiouts_llm_model(const bool with_pooled_output) {
        ov::test::npuw::LLMConfig cfg;
        cfg.add_hidden_states_output = true;
        cfg.hidden_states_output_name = "hidden_states";

        if (with_pooled_output) {
            cfg.add_pooled_output = true;
            cfg.pooled_output_name = "pooled_hidden_state";
        }

        ov::test::npuw::ModelBuilder builder;
        return builder.build_llm(cfg);
    }

protected:
    ov::Core core;
    SimpleLLMPipeline simple_llm;
    ov::AnyMap chunked_prefill_props;
    std::vector<int64_t> input_ids;
};

TEST_F(MultiOutsLLMSmokeBehaviorNPUW, NonLogitsOutputIsFullSizedAfterChunkedPrefill) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    const auto multiouts_llm = build_multiouts_llm_model(false);
    simple_llm.initialize(multiouts_llm, core, chunked_prefill_props);
    EXPECT_NO_THROW(simple_llm.generate(input_ids));

    const auto& llm_infer_request = simple_llm.get_request();
    const auto hidden_states = llm_infer_request->get_tensor("hidden_states");
    EXPECT_EQ(hidden_states.get_shape(), (ov::Shape{1, 128, 64}));

    const auto logits = llm_infer_request->get_tensor("logits");
    EXPECT_EQ(logits.get_shape()[0], 1u);
    EXPECT_EQ(logits.get_shape()[1], 1u);
}

TEST_F(MultiOutsLLMSmokeBehaviorNPUW, DifferentNonLogitsOutputs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    const auto multi_different_outs_llm = build_multiouts_llm_model(true);
    simple_llm.initialize(multi_different_outs_llm, core, chunked_prefill_props);
    EXPECT_NO_THROW(simple_llm.generate(input_ids));

    const auto& llm_infer_request = simple_llm.get_request();
    const auto hidden_states = llm_infer_request->get_tensor("hidden_states");
    EXPECT_EQ(hidden_states.get_shape(), (ov::Shape{1, 128, 64}));

    const auto logits = llm_infer_request->get_tensor("logits");
    EXPECT_EQ(logits.get_shape()[0], 1u);
    EXPECT_EQ(logits.get_shape()[1], 1u);

    const auto pooled_hidden = llm_infer_request->get_tensor("pooled_hidden_state");
    EXPECT_EQ(pooled_hidden.get_shape(), (ov::Shape{1, 64}));
    EXPECT_EQ(pooled_hidden.get_shape().size(), 2u);
}
