// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/npuw_private_properties.hpp"

namespace {

struct ConfigAssertionCase {
    std::string test_name;
    ::intel_npu::Config::ConfigMap options;
    std::function<void(const ::intel_npu::Config&)> verify;
};

using Case = ConfigAssertionCase;

std::shared_ptr<const ::intel_npu::OptionsDesc> make_options_desc() {
    auto desc = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*desc);
    ::intel_npu::registerNPUWLLMOptions(*desc);
    ::intel_npu::registerNPUWKokoroOptions(*desc);
    return desc;
}

::intel_npu::Config make_config(const ::intel_npu::Config::ConfigMap& options = {}) {
    ::intel_npu::Config cfg(make_options_desc());
    cfg.update(options);
    return cfg;
}

template <class Opt>
Case bool_case(std::string test_name, std::string value, bool expected) {
    return {std::move(test_name),
            {{std::string(Opt::key()), std::move(value)}},
            [expected](const ::intel_npu::Config& cfg) {
                EXPECT_EQ(cfg.get<Opt>(), expected);
            }};
}

template <class Opt>
Case string_case(std::string test_name, std::string value, std::string expected) {
    return {std::move(test_name),
            {{std::string(Opt::key()), std::move(value)}},
            [expected = std::move(expected)](const ::intel_npu::Config& cfg) {
                EXPECT_EQ(cfg.get<Opt>(), expected);
            }};
}

template <class Opt, class ValueT>
Case numeric_case(std::string test_name, std::string value, ValueT expected) {
    return {std::move(test_name),
            {{std::string(Opt::key()), std::move(value)}},
            [expected](const ::intel_npu::Config& cfg) {
                EXPECT_EQ(cfg.get<Opt>(), expected);
            }};
}

template <class Opt>
Case double_case(std::string test_name, std::string value, double expected) {
    return {std::move(test_name),
            {{std::string(Opt::key()), std::move(value)}},
            [expected](const ::intel_npu::Config& cfg) {
                EXPECT_DOUBLE_EQ(cfg.get<Opt>(), expected);
            }};
}

template <class Opt>
Case enum_case(std::string test_name, std::string value, std::string expected) {
    return {std::move(test_name),
            {{std::string(Opt::key()), std::move(value)}},
            [expected = std::move(expected)](const ::intel_npu::Config& cfg) {
                EXPECT_EQ(cfg.getString<Opt>(), expected);
            }};
}

std::vector<Case> make_cases() {
    std::vector<Case> cases = {
        bool_case<::intel_npu::NPU_USE_NPUW>("NPU_USE_NPUW", "YES", true),
        string_case<::intel_npu::NPUW_DEVICES>("NPUW_DEVICES", "CPU,NPU", "CPU,NPU"),
        string_case<::intel_npu::NPUW_SUBMODEL_DEVICE>("NPUW_SUBMODEL_DEVICE", "0:CPU,last:NPU", "0:CPU,last:NPU"),
        string_case<::intel_npu::NPUW_WEIGHTS_BANK>("NPUW_WEIGHTS_BANK", "shared-bank", "shared-bank"),
        string_case<::intel_npu::NPUW_WEIGHTS_BANK_ALLOC>("NPUW_WEIGHTS_BANK_ALLOC", "CPU", "CPU"),
        string_case<::intel_npu::NPUW_CACHE_DIR>("NPUW_CACHE_DIR", "/tmp/npuw-cache", "/tmp/npuw-cache"),
        string_case<::intel_npu::NPUW_ONLINE_PIPELINE>("NPUW_ONLINE_PIPELINE", "SPATIAL", "SPATIAL"),
        string_case<::intel_npu::NPUW_ONLINE_AVOID>("NPUW_ONLINE_AVOID", "Op:Select/NPU", "Op:Select/NPU"),
        string_case<::intel_npu::NPUW_ONLINE_ISOLATE>("NPUW_ONLINE_ISOLATE", "Op:Select/compute", "Op:Select/compute"),
        string_case<::intel_npu::NPUW_ONLINE_NO_FOLD>("NPUW_ONLINE_NO_FOLD", "compute", "compute"),
        numeric_case<::intel_npu::NPUW_ONLINE_MIN_SIZE>("NPUW_ONLINE_MIN_SIZE", "12", std::size_t{12}),
        numeric_case<::intel_npu::NPUW_ONLINE_KEEP_BLOCKS>("NPUW_ONLINE_KEEP_BLOCKS", "2", std::size_t{2}),
        numeric_case<::intel_npu::NPUW_ONLINE_KEEP_BLOCK_SIZE>("NPUW_ONLINE_KEEP_BLOCK_SIZE", "4", std::size_t{4}),
        string_case<::intel_npu::NPUW_ONLINE_DUMP_PLAN>("NPUW_ONLINE_DUMP_PLAN", "/tmp/plan.xml", "/tmp/plan.xml"),
        string_case<::intel_npu::NPUW_PLAN>("NPUW_PLAN", "/tmp/plan.xml", "/tmp/plan.xml"),
        bool_case<::intel_npu::NPUW_FOLD>("NPUW_FOLD", "YES", true),
        bool_case<::intel_npu::NPUW_CWAI>("NPUW_CWAI", "YES", true),
        bool_case<::intel_npu::NPUW_DQ>("NPUW_DQ", "YES", true),
        bool_case<::intel_npu::NPUW_DQ_FULL>("NPUW_DQ_FULL", "NO", false),
        string_case<::intel_npu::NPUW_PMM>("NPUW_PMM", "0,2", "0,2"),
        bool_case<::intel_npu::NPUW_MM_GATED>("NPUW_MM_GATED", "NO", false),
        bool_case<::intel_npu::NPUW_SLICE_OUT>("NPUW_SLICE_OUT", "YES", true),
        bool_case<::intel_npu::NPUW_F16IC>("NPUW_F16IC", "NO", false),
        string_case<::intel_npu::NPUW_DCOFF_TYPE>("NPUW_DCOFF_TYPE", "f16", "f16"),
        bool_case<::intel_npu::NPUW_DCOFF_SCALE>("NPUW_DCOFF_SCALE", "YES", true),
        bool_case<::intel_npu::NPUW_FUNCALL_FOR_ALL>("NPUW_FUNCALL_FOR_ALL", "YES", true),
        bool_case<::intel_npu::NPUW_HOST_GATHER>("NPUW_HOST_GATHER", "NO", false),
        bool_case<::intel_npu::NPUW_SPATIAL>("NPUW_SPATIAL", "YES", true),
        numeric_case<::intel_npu::NPUW_SPATIAL_NWAY>("NPUW_SPATIAL_NWAY", "16", std::size_t{16}),
        bool_case<::intel_npu::NPUW_SPATIAL_DYN>("NPUW_SPATIAL_DYN", "NO", false),
        numeric_case<::intel_npu::NPUW_MOE_TOKEN_CHUNK_SIZE>("NPUW_MOE_TOKEN_CHUNK_SIZE", "32", uint64_t{32}),
        numeric_case<::intel_npu::NPUW_MOE_POOL_SIZE>("NPUW_MOE_POOL_SIZE", "4", std::size_t{4}),
        string_case<::intel_npu::NPUW_ATTN>("NPUW_ATTN", "HFA", "HFA"),
        bool_case<::intel_npu::NPUW_ATTN_DYN>("NPUW_ATTN_DYN", "NO", false),
        bool_case<::intel_npu::NPUW_ATTN_NO_COPY>("NPUW_ATTN_NO_COPY", "YES", true),
        bool_case<::intel_npu::NPUW_ATTN_HFA_FUSED>("NPUW_ATTN_HFA_FUSED", "YES", true),
        bool_case<::intel_npu::NPUW_PARALLEL_COMPILE>("NPUW_PARALLEL_COMPILE", "YES", true),
        bool_case<::intel_npu::NPUW_FUNCALL_ASYNC>("NPUW_FUNCALL_ASYNC", "YES", true),
        bool_case<::intel_npu::NPUW_UNFOLD_IREQS>("NPUW_UNFOLD_IREQS", "YES", true),
        bool_case<::intel_npu::NPUW_FALLBACK_EXEC>("NPUW_FALLBACK_EXEC", "NO", false),
        bool_case<::intel_npu::NPUW_ACC_CHECK>("NPUW_ACC_CHECK", "YES", true),
        double_case<::intel_npu::NPUW_ACC_THRESH>("NPUW_ACC_THRESH", "0.25", 0.25),
        string_case<::intel_npu::NPUW_ACC_DEVICE>("NPUW_ACC_DEVICE", "CPU", "CPU"),
        bool_case<::intel_npu::NPUW_LLM>("NPUW_LLM", "YES", true),
        numeric_case<::intel_npu::NPUW_LLM_BATCH_DIM>("NPUW_LLM_BATCH_DIM", "1", uint32_t{1}),
        numeric_case<::intel_npu::NPUW_LLM_SEQ_LEN_DIM>("NPUW_LLM_SEQ_LEN_DIM", "3", uint32_t{3}),
        numeric_case<::intel_npu::NPUW_LLM_MAX_PROMPT_LEN>("NPUW_LLM_MAX_PROMPT_LEN", "2048", uint32_t{2048}),
        numeric_case<::intel_npu::NPUW_LLM_MAX_GENERATION_TOKEN_LEN>("NPUW_LLM_MAX_GENERATION_TOKEN_LEN",
                                                                     "8",
                                                                     uint32_t{8}),
        numeric_case<::intel_npu::NPUW_LLM_MIN_RESPONSE_LEN>("NPUW_LLM_MIN_RESPONSE_LEN", "192", uint32_t{192}),
        numeric_case<::intel_npu::NPUW_LLM_MAX_LORA_RANK>("NPUW_LLM_MAX_LORA_RANK", "64", uint32_t{64}),
        bool_case<::intel_npu::NPUW_LLM_OPTIMIZE_V_TENSORS>("NPUW_LLM_OPTIMIZE_V_TENSORS", "NO", false),
        bool_case<::intel_npu::NPUW_LLM_OPTIMIZE_FP8>("NPUW_LLM_OPTIMIZE_FP8", "YES", true),
        bool_case<::intel_npu::NPUW_LLM_CACHE_ROPE>("NPUW_LLM_CACHE_ROPE", "NO", false),
        enum_case<::intel_npu::NPUW_LLM_PREFILL_MOE_HINT>("NPUW_LLM_PREFILL_MOE_HINT", "HOST_ROUTED", "HOST_ROUTED"),
        enum_case<::intel_npu::NPUW_LLM_GENERATE_MOE_HINT>("NPUW_LLM_GENERATE_MOE_HINT",
                                                           "DEVICE_ROUTED",
                                                           "DEVICE_ROUTED"),
        bool_case<::intel_npu::NPUW_LLM_GENERATE_PYRAMID>("NPUW_LLM_GENERATE_PYRAMID", "YES", true),
        numeric_case<::intel_npu::NPUW_LLM_PREFILL_CHUNK_SIZE>("NPUW_LLM_PREFILL_CHUNK_SIZE", "256", uint64_t{256}),
        bool_case<::intel_npu::NPUW_LLM_ENABLE_PREFIX_CACHING>("NPUW_LLM_ENABLE_PREFIX_CACHING", "YES", true),
        numeric_case<::intel_npu::NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE>("NPUW_LLM_PREFIX_CACHING_BLOCK_SIZE",
                                                                      "128",
                                                                      uint64_t{128}),
        numeric_case<::intel_npu::NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS>(
            "NPUW_LLM_PREFIX_CACHING_MAX_NUM_BLOCKS",
            "32",
            uint64_t{32}),
        enum_case<::intel_npu::NPUW_LLM_PREFILL_HINT>("NPUW_LLM_PREFILL_HINT", "STATIC", "STATIC"),
        enum_case<::intel_npu::NPUW_LLM_PREFILL_ATTENTION_HINT>("NPUW_LLM_PREFILL_ATTENTION_HINT",
                                                                "DYNAMIC",
                                                                "DYNAMIC"),
        enum_case<::intel_npu::NPUW_LLM_GENERATE_HINT>("NPUW_LLM_GENERATE_HINT", "BEST_PERF", "BEST_PERF"),
        enum_case<::intel_npu::NPUW_LLM_GENERATE_ATTENTION_HINT>("NPUW_LLM_GENERATE_ATTENTION_HINT",
                                                                 "PYRAMID",
                                                                 "PYRAMID"),
        bool_case<::intel_npu::NPUW_LLM_SHARED_HEAD>("NPUW_LLM_SHARED_HEAD", "NO", false),
        bool_case<::intel_npu::NPUW_WHISPER>("NPUW_WHISPER", "YES", true),
        numeric_case<::intel_npu::NPUW_WHISPER_EOS_TOKEN>("NPUW_WHISPER_EOS_TOKEN", "42", uint64_t{42}),
        bool_case<::intel_npu::NPUW_EAGLE>("NPUW_EAGLE", "YES", true),
        bool_case<::intel_npu::NPUW_TEXT_EMBED>("NPUW_TEXT_EMBED", "YES", true),
        bool_case<::intel_npu::NPUW_KOKORO>("NPUW_KOKORO", "YES", true),
        numeric_case<::intel_npu::NPUW_KOKORO_BLOCK_SIZE>("NPUW_KOKORO_BLOCK_SIZE", "256", uint64_t{256}),
        numeric_case<::intel_npu::NPUW_KOKORO_OVERLAP_SIZE>("NPUW_KOKORO_OVERLAP_SIZE", "32", uint64_t{32}),
        {"NPUW_WEIGHTS_HANDLE_PROVIDER",
         {},
         [](const ::intel_npu::Config&) {
             EXPECT_EQ(std::string(ov::intel_npu::npuw::weights_handle_provider.name()),
                       "NPUW_WEIGHTS_HANDLE_PROVIDER");
         }},
    };

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    cases.push_back(bool_case<::intel_npu::NPUW_DUMP_FULL>("NPUW_DUMP_FULL", "YES", true));
    cases.push_back(string_case<::intel_npu::NPUW_DUMP_SUBS>("NPUW_DUMP_SUBS", "YES", "YES"));
    cases.push_back(string_case<::intel_npu::NPUW_DUMP_SUBS_DIR>("NPUW_DUMP_SUBS_DIR", "/tmp/dumps", "/tmp/dumps"));
    cases.push_back(string_case<::intel_npu::NPUW_DUMP_SUBS_ON_FAIL>("NPUW_DUMP_SUBS_ON_FAIL", "last", "last"));
    cases.push_back(string_case<::intel_npu::NPUW_DUMP_IO>("NPUW_DUMP_IO", "0,last", "0,last"));
    cases.push_back(bool_case<::intel_npu::NPUW_DUMP_IO_ITERS>("NPUW_DUMP_IO_ITERS", "YES", true));
#endif

    return cases;
}

class SmokeTest : public ::testing::TestWithParam<Case> {};

TEST_P(SmokeTest, ParsesDocumentedOption) {
    auto cfg = make_config(GetParam().options);
    GetParam().verify(cfg);
}

std::string case_name(const testing::TestParamInfo<Case>& info) {
    return info.param.test_name;
}

INSTANTIATE_TEST_SUITE_P(NPUWOptions, SmokeTest, ::testing::ValuesIn(make_cases()), case_name);

TEST(NPUWConfigOptionsSmokeTest, AttentionHintDefaultsCanDifferPerOption) {
    const auto cfg = make_config();

    EXPECT_EQ(cfg.getString<::intel_npu::NPUW_LLM_PREFILL_ATTENTION_HINT>(), "PYRAMID");
    EXPECT_EQ(cfg.getString<::intel_npu::NPUW_LLM_GENERATE_ATTENTION_HINT>(), "STATIC");
}

}  // namespace
