// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "llm_pass_test_fixture.hpp"
#include "openvino/runtime/properties.hpp"

// ─── Design note ─────────────────────────────────────────────────────────────
// The model builder creates KV cache state with ov::element::f32 (the default
// ModelConfig::precision).  ConvertKVCacheToPrecision is therefore doing *real*
// work on the test model: it lowers f32 past_key inputs and present outputs to
// the requested storage type (f16 by default, or whatever
// ov::hint::kv_cache_precision selects).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using ov::test::npuw::RecordingFactory;

// Returns true if at least one input whose name contains `needle` exists AND
// every such input has `expected_type`.
static bool all_inputs_with_name_have_type(const std::shared_ptr<ov::Model>& model,
                                           const std::string& needle,
                                           ov::element::Type expected_type) {
    bool found_any = false;
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            if (name.find(needle) != std::string::npos) {
                found_any = true;
                if (input.get_element_type() != expected_type)
                    return false;
                break;
            }
        }
    }
    return found_any;
}

// Returns true if at least one output whose name contains `needle` exists AND
// every such output has `expected_type`.
static bool all_outputs_with_name_have_type(const std::shared_ptr<ov::Model>& model,
                                            const std::string& needle,
                                            ov::element::Type expected_type) {
    bool found_any = false;
    for (const auto& output : model->outputs()) {
        for (const auto& name : output.get_names()) {
            if (name.find(needle) != std::string::npos) {
                found_any = true;
                if (output.get_element_type() != expected_type)
                    return false;
                break;
            }
        }
    }
    return found_any;
}

// Returns true if NO input whose name contains `needle` has `excluded_type`.
static bool no_inputs_with_name_have_type(const std::shared_ptr<ov::Model>& model,
                                          const std::string& needle,
                                          ov::element::Type excluded_type) {
    for (const auto& input : model->inputs()) {
        for (const auto& name : input.get_names()) {
            if (name.find(needle) != std::string::npos) {
                if (input.get_element_type() == excluded_type)
                    return false;
                break;
            }
        }
    }
    return true;
}

// ─── Parametrized fixture ─────────────────────────────────────────────────────
// Parametrized over ov::element::Type so that f16, f8e4m3, and f8e5m2 are each
// exercised in exactly the same test bodies.

class ConvertKVCacheHintPrecisionTest : public ov::test::npuw::LLMPassTestFixture,
                                        public ::testing::WithParamInterface<ov::element::Type> {};

INSTANTIATE_TEST_SUITE_P(
    KVCachePrecisions,
    ConvertKVCacheHintPrecisionTest,
    ::testing::Values(ov::element::f16, ov::element::f8e4m3, ov::element::f8e5m2),
    [](const ::testing::TestParamInfo<ov::element::Type>& info) -> std::string {
        std::ostringstream ss;
        ss << info.param;
        return ss.str();
    });

// past_key inputs of the generate model have the requested precision.
TEST_P(ConvertKVCacheHintPrecisionTest, GenerateModelPastKeyInputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        {{ov::hint::kv_cache_precision.name(), kv_type}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    EXPECT_TRUE(all_inputs_with_name_have_type(generate->model, "past_key", kv_type))
        << "past_key inputs must have type " << kv_type;
}

// present outputs of the generate model have the requested precision.
TEST_P(ConvertKVCacheHintPrecisionTest, GenerateModelPresentOutputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        {{ov::hint::kv_cache_precision.name(), kv_type}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    EXPECT_TRUE(all_outputs_with_name_have_type(generate->model, "present", kv_type))
        << "present outputs must have type " << kv_type;
}

// present outputs of the prefill model have the requested precision.
TEST_P(ConvertKVCacheHintPrecisionTest, PrefillModelPresentOutputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(
                        {{ov::hint::kv_cache_precision.name(), kv_type}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);

    EXPECT_TRUE(all_outputs_with_name_have_type(prefill->model, "present", kv_type))
        << "present outputs in the prefill model must have type " << kv_type;
}

// ─── Non-parametric tests ─────────────────────────────────────────────────────

class ConvertKVCacheToPrecisionPassTest : public ov::test::npuw::LLMPassTestFixture {};

// Chunked-prefill: past_key inputs of the prefill model are converted to f16.
TEST_F(ConvertKVCacheToPrecisionPassTest, ChunkedPrefillModelPastKeyInputsAreF16) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);

    EXPECT_TRUE(all_inputs_with_name_have_type(prefill->model, "past_key", ov::element::f16))
        << "past_key inputs in chunked prefill must be f16 after ConvertKVCacheToPrecision";
}

// Non-KV inputs (input_ids) must not be touched by ConvertKVCacheToPrecision.
TEST_F(ConvertKVCacheToPrecisionPassTest, NonKVInputsAreNotConverted) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(generate, nullptr);

    EXPECT_TRUE(no_inputs_with_name_have_type(generate->model, "input_ids", ov::element::f16))
        << "input_ids must NOT be f16 — ConvertKVCacheToPrecision must not touch it";
}

}  // namespace
