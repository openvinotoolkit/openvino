// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_engine/comparators/nrmse.hpp"
#include "test_engine/simple_llm_pipeline.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "openvino/util/shared_object.hpp"
#include "openvino/util/file_util.hpp"

#include <filesystem>
#include <algorithm>

#include <gtest/gtest.h>

using namespace testing;
using namespace ov::npuw::tests;
using namespace ov::intel_npu::npuw;

// this tests load plugin by library name: this is not available during static linkage
#ifndef OPENVINO_STATIC_LIBRARY
#ifdef WITH_CPU_PLUGIN
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
const char* cpu_plugin_file_name = "openvino_intel_cpu_plugin";

namespace {
const std::vector<int64_t> What_is_OpenVINO =
    {529, 29989, 1792, 29989, 29958, 13, 5618, 338, 4673, 29963, 1177,
     29949, 29973, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13};
const std::vector<int64_t> OpenVINO =
    {6585, 29963, 1177, 29949}; 

using LLMTestParams = std::tuple<std::string, ov::AnyMap, 
                                 std::vector<int64_t>, std::vector<int64_t>>;
} // anonymous namespace

class LLMAccuracyTestsNPUW : public ::testing::TestWithParam<LLMTestParams> {
public:
    void SetUp() override {
        // NOTE: TEMPLATE plugin in OpenVINO works for ~20 minute to generate
        //       first token from prefill model and crashes on launch of
        //       3rd subrequest in generate model.
        //       There is no such issue with CPU plugin, so CPU plugin was choosen
        //       for accuracy checks.
        // Register CPU plugin in OpenVINO:
        try {
            core.register_plugin(std::string(cpu_plugin_file_name) + OV_BUILD_POSTFIX, "CPU");
        } catch (ov::Exception& ex) {
            if (std::string{ex.what()}.find("Device with \"CPU\"  is already registered in the OpenVINO Runtime")
                == std::string::npos) {
                throw ex;
            }
        }

        auto param = GetParam();
        ov::AnyMap config;
        std::tie(model_path, config, input_ids, reference_ids) = param;
        config["NPUW_DEVICES"] = "CPU";
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

TEST_P(LLMAccuracyTestsNPUW, ConfigIsAccurate) {
    actual_ids = simple_llm.generate(input_ids);
    for (auto i = 0; i < actual_ids.size(); ++i) {
        ASSERT_EQ(actual_ids[i], reference_ids[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(LLMAccuracyNPUW_FAST_COMPILE, LLMAccuracyTestsNPUW,
    ::testing::Combine(testing::Values("C:\\apronina\\models\\TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful\\TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful\\openvino_model.xml"),
                       testing::Values(ov::AnyMap{}),
                       testing::Values(What_is_OpenVINO),
                       testing::Values(OpenVINO)));

INSTANTIATE_TEST_SUITE_P(LLMAccuracyNPUW_BEST_PERF, LLMAccuracyTestsNPUW,
::testing::Combine(testing::Values("C:\\apronina\\models\\TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful\\TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful\\openvino_model.xml"),
                    testing::Values(ov::AnyMap{{"NPUW_LLM_GENERATE_HINT", "BEST_PERF"}}),
                    testing::Values(What_is_OpenVINO),
                    testing::Values(OpenVINO)));

INSTANTIATE_TEST_SUITE_P(LLMAccuracyNPUW_DYNAMIC_BEST_PERF, LLMAccuracyTestsNPUW,
    ::testing::Combine(testing::Values("C:\\apronina\\models\\TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful\\TinyLlama-1.1B-Chat-v1.0_int4_sym_group128_dyn_stateful\\openvino_model.xml"),
                        testing::Values(ov::AnyMap{{"NPUW_LLM_GENERATE_HINT", "BEST_PERF"},
                                                   {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}}),
                        testing::Values(What_is_OpenVINO),
                        testing::Values(OpenVINO)));

#endif // defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#endif // WITH_CPU_PLUGIN
#endif // not OPENVINO_STATIC_LIBRARY
