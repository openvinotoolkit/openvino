// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include "functional_test_utils/skip_tests_config.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "test_engine/comparators/nrmse.hpp"
#include "test_engine/models/find_model.hpp"
#include "test_engine/simple_llm_pipeline.hpp"

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

namespace {
const std::string minicpm_05_b_name = "MiniCPM4-0.5B_int4_sym_group128_dyn_stateful";

struct GroundTruth {
    std::vector<int64_t> prompt;
    std::vector<int64_t> answer;
};

const GroundTruth What_is_OpenVINO_templated = {
    // Prompt:
    // <|im_start|>user
    // What is OpenVINO?<|im_end|>
    // <|im_start|>assistant
    {73441, 3060, 5, 5856, 1410, 6404, 59408, 2097, 59383, 74, 73440, 59320, 5, 73441, 16434, 5},
    // Answer: OpenVINO
    { 9254, 59408, 2097, 59383}
};

using LLMTestParams = std::tuple<std::string, ov::AnyMap, GroundTruth>;
} // anonymous namespace

class LLMSmokeAccuracyTestsNPUW : public ::testing::TestWithParam<LLMTestParams> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        // NOTE: TEMPLATE plugin in OpenVINO works for ~20 minute to generate
        //       first token from prefill model and crashes on launch of
        //       3rd subrequest in generate model.
        //       There is no such issue with CPU plugin, so CPU plugin was choosen
        //       for accuracy checks.
        // Test only makes sense if CPU plugin is enabled.
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
       // (defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64))

        auto param = GetParam();
        std::string model_name;
        ov::AnyMap config;
        GroundTruth input_and_reference_ids;
        std::tie(model_name, config, input_and_reference_ids) = param;

        model_path = find_model(model_name);
        if (model_path == "") {
            GTEST_SKIP() << "Test model is not found, skipping the test!";
        }

        input_ids = input_and_reference_ids.prompt;
        reference_ids = input_and_reference_ids.answer;

        config["NPUW_DEVICES"] = "CPU";
        config["NPUW_LLM_MAX_PROMPT_LEN"] = 128;
        config["NPUW_LLM_MIN_RESPONSE_LEN"] = 4;
        simple_llm.initialize(model_path, core, config);
    }

protected:
    ov::Core core;
    SimpleLLMPipeline simple_llm;
    std::string model_path;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> actual_ids;
    std::vector<int64_t> reference_ids;
};

TEST_P(LLMSmokeAccuracyTestsNPUW, ConfigIsAccurate) {
    actual_ids = simple_llm.generate(input_ids);
    for (std::size_t i = 0; i < actual_ids.size(); ++i) {
        ASSERT_EQ(actual_ids[i], reference_ids[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(LLMSmokeAccuracyNPUW_FAST_COMPILE, LLMSmokeAccuracyTestsNPUW,
    ::testing::Combine(testing::Values(minicpm_05_b_name),
                       testing::Values(ov::AnyMap{}),
                       testing::Values(What_is_OpenVINO_templated)));
