// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <sstream>
#include <string>

#include "llm_pass_test_fixture.hpp"
#include "../util.hpp"
#include "npuw_transformations/convert_kvcache_to_precision.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/properties.hpp"
#include "whisper/prepare_whisper_model.hpp"

// --- Design note -------------------------------------------------------------------------
// The model builder creates KV cache state with ov::element::f32 (the default
// BaseModelConfig::precision).  ConvertKVCacheToPrecision is therefore doing *real*
// work on the test model: it lowers f32 past_key inputs and present outputs to
// the requested storage type (f16 by default, or whatever
// ov::hint::kv_cache_precision selects).
// -----------------------------------------------------------------------------------------

namespace {

using ov::test::npuw::RecordingFactory;

bool any_name_contains(const ov::Output<const ov::Node>& port, std::string_view needle) {
    for (const auto& name : port.get_names()) {
        if (name.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

const std::map<ov::element::Type, std::map<std::string, ov::element::Type>>& precision_key_matrix() {
    static const std::map<ov::element::Type, std::map<std::string, ov::element::Type>> matrix = {
        {ov::element::u8, {{"value", ov::element::u8}, {"scale", ov::element::f32}, {"zero_point", ov::element::u8}}},
        {ov::element::i8, {{"value", ov::element::i8}, {"scale", ov::element::f32}, {"zero_point", ov::element::i8}}}
    };
    return matrix;
}

const std::map<ov::element::Type, std::map<std::string, ov::element::Type>>& precision_value_matrix() {
    static const std::map<ov::element::Type, std::map<std::string, ov::element::Type>> matrix = {
        {ov::element::u8, {{"value", ov::element::i8}, {"scale", ov::element::f32}}},
        {ov::element::i8, {{"value", ov::element::i8}, {"scale", ov::element::f32}}}
    };
    return matrix;
}

bool is_quantized_kv_type(const ov::element::Type kv_type) {
    return precision_key_matrix().count(kv_type) > 0;
}

bool is_fp8_kv_type(const ov::element::Type kv_type) {
    return kv_type == ov::element::f8e4m3 || kv_type == ov::element::f8e5m2 || kv_type == ov::element::f8e8m0;
}

ov::AnyMap make_kv_precision_props(const ov::element::Type kv_type) {
    ov::AnyMap props = {{ov::hint::kv_cache_precision.name(), kv_type}};
    if (is_fp8_kv_type(kv_type)) {
        props["NPUW_LLM_OPTIMIZE_FP8"] = "YES";
    }
    return props;
}

void expect_kv_cache_input_types(const std::shared_ptr<ov::Model>& model,
                                 const ov::element::Type kv_type,
                                 const bool check_quant_aux_ports = true) {
    // Key cache: asymmetric quantization -> value tensor + scale (f32) + zero_point (same as quant type).
    // Value cache: symmetric quantization -> value tensor (i4) + scale (f32), no zero_point.
    const bool is_quantized = is_quantized_kv_type(kv_type);

    constexpr std::string_view past_key_scale_name = "/past_key_values/key/scale";
    constexpr std::string_view past_key_zp_name = "/past_key_values/key/zp";
    constexpr std::string_view past_value_scale_name = "/past_key_values/value/scale";
    constexpr std::string_view past_value_zp_name = "/past_key_values/value/zp";

    bool found_key_cache_input = false;
    bool found_value_cache_input = false;
    bool found_key_scale_input = false;
    bool found_key_zp_input = false;
    bool found_value_scale_input = false;
    bool found_value_zp_input = false;

    for (const auto& input : model->inputs()) {
        // Check if any name on this input matches past_key_values pattern
        bool is_past_key = false;
        bool is_past_value = false;
        for (const auto& name : input.get_names()) {
            if (!is_past_key && ov::npuw::util::isPastKeyValuesKey(name).has_value()) {
                is_past_key = true;
            }
            if (!is_past_value && ov::npuw::util::isPastKeyValuesValue(name).has_value()) {
                is_past_value = true;
            }
        }

        if (is_past_key) {
            found_key_cache_input = true;
            const auto expected = is_quantized ? precision_key_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_key_values.<N>.key input must have type " << expected;
        }

        if (is_past_value) {
            found_value_cache_input = true;
            const auto expected = is_quantized ? precision_value_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_key_values.<N>.value input must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(input, past_key_scale_name)) {
            found_key_scale_input = true;
            const auto expected = precision_key_matrix().at(kv_type).at("scale");
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_key scale input must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(input, past_key_zp_name)) {
            found_key_zp_input = true;
            const auto expected = precision_key_matrix().at(kv_type).at("zero_point");
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_key zero-point input must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(input, past_value_scale_name)) {
            found_value_scale_input = true;
            const auto expected = precision_value_matrix().at(kv_type).at("scale");
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_value scale input must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(input, past_value_zp_name)) {
            found_value_zp_input = true;
        }
    }

    EXPECT_TRUE(found_key_cache_input) << "No past_key_values.<N>.key input found in model";
    EXPECT_TRUE(found_value_cache_input) << "No past_key_values.<N>.value input found in model";

    if (is_quantized && check_quant_aux_ports) {
        EXPECT_TRUE(found_key_scale_input)
            << "Asymmetric quantized KV key-cache must expose scale input";
        EXPECT_TRUE(found_key_zp_input)
            << "Asymmetric quantized KV key-cache must expose zero-point input";
        EXPECT_TRUE(found_value_scale_input)
            << "Symmetric quantized KV value-cache must expose scale input";
        EXPECT_FALSE(found_value_zp_input)
            << "Symmetric quantized KV value-cache must not expose zero-point input";
    } else if (!is_quantized && check_quant_aux_ports) {
        EXPECT_FALSE(found_key_scale_input) << "Non-quantized KV-cache must not expose key scale input";
        EXPECT_FALSE(found_key_zp_input) << "Non-quantized KV-cache must not expose key zero-point input";
        EXPECT_FALSE(found_value_scale_input) << "Non-quantized KV-cache must not expose value scale input";
        EXPECT_FALSE(found_value_zp_input) << "Non-quantized KV-cache must not expose value zero-point input";
    }
}

void expect_kv_cache_present_output_types(const std::shared_ptr<ov::Model>& model,
                                          const ov::element::Type kv_type,
                                          const bool check_quant_aux_ports = true) {
    const bool is_quantized = is_quantized_kv_type(kv_type);

    constexpr std::string_view present_key_scale_name = "/present/key/scale";
    constexpr std::string_view present_key_zp_name = "/present/key/zp";
    constexpr std::string_view present_value_scale_name = "/present/value/scale";
    constexpr std::string_view present_value_zp_name = "/present/value/zp";

    bool found_present_key = false;
    bool found_present_value = false;
    bool found_present_key_scale = false;
    bool found_present_key_zp = false;
    bool found_present_value_scale = false;
    bool found_present_value_zp = false;

    for (const auto& output : model->outputs()) {
        // Check if any name on this output matches present pattern
        bool is_present_key = false;
        bool is_present_value = false;
        for (const auto& name : output.get_names()) {
            if (!is_present_key && ov::npuw::util::isPresentKeyValuesKey(name).has_value()) {
                is_present_key = true;
            }
            if (!is_present_value && ov::npuw::util::isPresentKeyValuesValue(name).has_value()) {
                is_present_value = true;
            }
        }

        if (is_present_key) {
            found_present_key = true;
            const auto expected = is_quantized ? precision_key_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(output.get_element_type(), expected)
                << "present.<N>.key output must have type " << expected;
        }

        if (is_present_value) {
            found_present_value = true;
            const auto expected = is_quantized ? precision_value_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(output.get_element_type(), expected)
                << "present.<N>.value output must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(output, present_key_scale_name)) {
            found_present_key_scale = true;
            const auto expected = precision_key_matrix().at(kv_type).at("scale");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present key scale output must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(output, present_key_zp_name)) {
            found_present_key_zp = true;
            const auto expected = precision_key_matrix().at(kv_type).at("zero_point");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present key zero-point output must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(output, present_value_scale_name)) {
            found_present_value_scale = true;
            const auto expected = precision_value_matrix().at(kv_type).at("scale");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present value scale output must have type " << expected;
        }

        if (check_quant_aux_ports && any_name_contains(output, present_value_zp_name)) {
            found_present_value_zp = true;
        }
    }

    EXPECT_TRUE(found_present_key) << "No present.<N>.key output found in model";
    EXPECT_TRUE(found_present_value) << "No present.<N>.value output found in model";

    if (is_quantized && check_quant_aux_ports) {
        EXPECT_TRUE(found_present_key_scale)
            << "Asymmetric quantized KV key-cache must expose present scale output";
        EXPECT_TRUE(found_present_key_zp)
            << "Asymmetric quantized KV key-cache must expose present zero-point output";
        EXPECT_TRUE(found_present_value_scale)
            << "Symmetric quantized KV value-cache must expose present scale output";
        EXPECT_FALSE(found_present_value_zp)
            << "Symmetric quantized KV value-cache must not expose present zero-point output";
    } else if (!is_quantized && check_quant_aux_ports) {
        EXPECT_FALSE(found_present_key_scale)
            << "Non-quantized KV-cache must not expose present key scale output";
        EXPECT_FALSE(found_present_key_zp)
            << "Non-quantized KV-cache must not expose present key zero-point output";
        EXPECT_FALSE(found_present_value_scale)
            << "Non-quantized KV-cache must not expose present value scale output";
        EXPECT_FALSE(found_present_value_zp)
            << "Non-quantized KV-cache must not expose present value zero-point output";
    }
}

// --- Parametrized fixture -------------------------------------------------------------------------
// Parametrized over ov::element::Type so that f16, f8e4m3, f8e5m2, i8, and u8 are each
// exercised in exactly the same test bodies.

class ConvertKVCacheHintPrecisionTest : public ov::test::npuw::LLMPassTestFixture,
                                        public ::testing::WithParamInterface<ov::element::Type> {};

INSTANTIATE_TEST_SUITE_P(
    KVCachePrecisions,
    ConvertKVCacheHintPrecisionTest,
    ::testing::Values(ov::element::f16,
                      ov::element::f8e4m3,
                      ov::element::f8e5m2,
                      ov::element::i8,
                      ov::element::u8),
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

    ASSERT_NO_THROW(compiled = create_compiled_model(make_kv_precision_props(kv_type), recorder));
    ASSERT_NE(compiled, nullptr);
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    expect_kv_cache_input_types(generate.model, kv_type);
}

// present outputs of the generate model have the requested precision.
TEST_P(ConvertKVCacheHintPrecisionTest, GenerateModelPresentOutputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(make_kv_precision_props(kv_type), recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    expect_kv_cache_present_output_types(generate.model, kv_type);
}

// present outputs of the prefill model have the requested precision.
TEST_P(ConvertKVCacheHintPrecisionTest, PrefillModelPresentOutputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(make_kv_precision_props(kv_type), recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");

    expect_kv_cache_present_output_types(prefill.model, kv_type);
}

// Whisper decoder_with_past model uses names like:
//   past_key_values.<idx>.decoder.key / present.<idx>.decoder.key
//   past_key_values.<idx>.encoder.key / present.<idx>.encoder.key
// Ensure KV-cache precision conversion handles those variants too.
TEST_P(ConvertKVCacheHintPrecisionTest, WhisperKVCacheModelPastKeyInputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    auto model = ov::test::npuw::build_whisper_decoder_test_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();
    ASSERT_TRUE(ov::npuw::util::PrepareWhisperKVCacheModel().run_on_model(model));
    ASSERT_TRUE(ov::npuw::ConvertKVCacheToPrecision(kv_type).run_on_model(model));

    expect_kv_cache_input_types(model, kv_type, false);
}

TEST_P(ConvertKVCacheHintPrecisionTest, WhisperKVCacheModelPresentOutputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    auto model = ov::test::npuw::build_whisper_decoder_test_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();
    ASSERT_TRUE(ov::npuw::util::PrepareWhisperKVCacheModel().run_on_model(model));
    ASSERT_TRUE(ov::npuw::ConvertKVCacheToPrecision(kv_type).run_on_model(model));

    expect_kv_cache_present_output_types(model, kv_type, false);
}

// --- Non-parametric tests -------------------------------------------------------------------------

class ConvertKVCacheToPrecisionPassTest : public ov::test::npuw::LLMPassTestFixture {};

// NPUW_LLM_OPTIMIZE_FP8: model with two consecutive FakeConvert nodes per K/V path.
// optimize_kv_cache_storage detects the FakeConvert destination type and sets KV storage to FP8.
// Uses non chunked prefill so the pipeline also tests the RedirectNewKvToOutput path (no down-up-proj needed).
TEST_F(ConvertKVCacheToPrecisionPassTest, OptimizeFp8ConsecutiveFakeConvertsKvCacheToFp8) {
    for (const auto fp8_type : {ov::element::f8e4m3, ov::element::f8e5m2}) {
        SCOPED_TRACE(std::string("optimize_fp8 type=") + fp8_type.get_type_name());

        RecordingFactory recorder;
        auto model = ov::test::npuw::build_llm_test_model_with_kv_fake_convert(fp8_type);
        std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
        try {
            compiled = create_compiled_model(model,
                                             {{"NPUW_LLM_OPTIMIZE_FP8", "YES"},
                                              {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                              {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}},
                                             recorder);
        } catch (const std::exception& ex) {
            FAIL() << "create_compiled_model failed with exception: " << ex.what();
        } catch (...) {
            FAIL() << "create_compiled_model failed with a non-std exception";
        }
        ASSERT_NE(compiled, nullptr);

        const auto& generate = require_sub_model_containing(recorder, "_kv");
        const auto& prefill = require_sub_model(recorder, "_prefill");

        expect_kv_cache_input_types(generate.model, fp8_type);
        expect_kv_cache_present_output_types(generate.model, fp8_type);
        expect_kv_cache_present_output_types(prefill.model, fp8_type);
    }
}

// NPUW_LLM_OPTIMIZE_FP8 should leave KV cache in f16 when the model has no suitable FakeConvert pattern.
TEST_F(ConvertKVCacheToPrecisionPassTest, OptimizeFp8WithPlainModelKeepsF16KvCache) {
    RecordingFactory recorder;
    ASSERT_NO_THROW(create_compiled_model({{"NPUW_LLM_OPTIMIZE_FP8", "YES"}}, recorder));

    const auto& generate = require_sub_model_containing(recorder, "_kv");
    const auto& prefill = require_sub_model(recorder, "_prefill");

    expect_kv_cache_input_types(generate.model, ov::element::f16);
    expect_kv_cache_present_output_types(generate.model, ov::element::f16);
    expect_kv_cache_present_output_types(prefill.model, ov::element::f16);
}


// Chunked-prefill: past_key inputs of the prefill model are converted to f16.
TEST_F(ConvertKVCacheToPrecisionPassTest, ChunkedPrefillModelPastKeyInputsAreF16) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");

    EXPECT_TRUE(all_inputs_with_name_have_type(prefill.model, "past_key", ov::element::f16))
        << "past_key inputs in chunked prefill must be f16 after ConvertKVCacheToPrecision";
}

// Non-KV inputs (input_ids) must not be touched by ConvertKVCacheToPrecision.
TEST_F(ConvertKVCacheToPrecisionPassTest, NonKVInputsAreNotConverted) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(no_inputs_with_name_have_type(generate.model, "input_ids", ov::element::f16))
        << "input_ids must NOT be f16 -- ConvertKVCacheToPrecision must not touch it";
}

}  // namespace
