// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "llm_pass_test_fixture.hpp"
#include "../util.hpp"
#include "infer_request_utils.hpp"
#include "llm_infer_request.hpp"
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

std::optional<std::string> resolve_kv_input_name_for_test(const std::string& output_name,
                                                           const std::unordered_set<std::string>& input_names) {
    auto input_name = ov::npuw::util::present_to_past_key_values_name(output_name);
    if (input_names.find(input_name) != input_names.end()) {
        return input_name;
    }

    const auto marker = std::string(ov::npuw::util::constants::past_key_values);
    const auto marker_pos = input_name.find(marker);
    if (marker_pos == std::string::npos) {
        return std::nullopt;
    }

    auto canonical_name = input_name.substr(marker_pos);
    if (input_names.find(canonical_name) != input_names.end()) {
        return canonical_name;
    }

    return std::nullopt;
}

class TestableLLMInferRequest final : public ov::npuw::LLMInferRequest {
public:
    explicit TestableLLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model)
        : ov::npuw::LLMInferRequest(compiled_model) {}

    using ov::npuw::LLMInferRequest::copy_kvcache;

    void prepare_non_chunked_copy() {
        auto& desc = ov::npuw::LLMInferRequest::kvcache_desc();
        ASSERT_FALSE(use_chunk_prefill());
        ASSERT_GT(desc.max_prompt_size, 0u);
        desc.num_stored_tokens = desc.max_prompt_size;
    }

    const ov::npuw::LLMCompiledModel::KVCacheDesc& kvcache_desc() const {
        return ov::npuw::LLMInferRequest::kvcache_desc();
    }

    const ov::npuw::LLMInferBaseRequest::PortsMap& prefill_in_ports() const {
        return m_prefill_in_ports;
    }

    const ov::npuw::LLMInferBaseRequest::PortsMap& prefill_out_ports() const {
        return m_prefill_out_ports;
    }

    const ov::npuw::LLMInferBaseRequest::PortsMap& kvcache_in_ports() const {
        return m_kvcache_in_ports;
    }

    const ov::npuw::LLMInferBaseRequest::PortsMap& kvcache_out_ports() const {
        return m_kvcache_out_ports;
    }

    std::shared_ptr<ov::IAsyncInferRequest> prefill_request() const {
        return m_prefill_request;
    }

    std::shared_ptr<ov::IAsyncInferRequest> kvcache_request() const {
        return m_kvcache_request;
    }
};

bool is_kv_name(std::string_view name) {
    return name.find(ov::npuw::util::constants::present) != std::string_view::npos ||
           name.find(ov::npuw::util::constants::past_key_values) != std::string_view::npos;
}

std::pair<ov::SoPtr<ov::ITensor>, ov::SoPtr<ov::ITensor>> make_non_chunked_copy_views(
    const TestableLLMInferRequest& request,
    const std::string& output_name,
    const ov::SoPtr<ov::ITensor>& src_tensor,
    const ov::SoPtr<ov::ITensor>& dst_tensor) {
    const auto& desc = request.kvcache_desc();
    const auto is_value_tensor = output_name.find("value") != std::string::npos;
    const auto kv_dim = [&](bool v_transposed) {
        return (is_value_tensor && v_transposed) ? 3u : desc.dim;
    };

    const auto pre_kv_dim = kv_dim(desc.v_tensors_transposed_pre);
    const auto gen_kv_dim = kv_dim(desc.v_tensors_transposed_gen);

    auto src_view = ov::npuw::util::make_tensor_slice(src_tensor,
                                                      pre_kv_dim,
                                                      desc.max_prompt_size - desc.num_stored_tokens,
                                                      desc.max_prompt_size);
    auto dst_view = ov::npuw::util::make_tensor_slice(dst_tensor, gen_kv_dim, 0u, desc.num_stored_tokens);
    return {src_view, dst_view};
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
                                 const bool ignore_quant_aux_ports = false) {
    // Key cache: asymmetric quantization -> value tensor + scale (f32) + zero_point (same as quant type).
    // Value cache: symmetric quantization -> value tensor (i4) + scale (f32), no zero_point.
    const bool is_quantized = is_quantized_kv_type(kv_type);

    const std::string past_key_scale_name =
        std::string("/") + ov::npuw::util::constants::past_key_values + "/key/scale";
    const std::string past_key_zp_name =
        std::string("/") + ov::npuw::util::constants::past_key_values + "/key/zp";
    const std::string past_value_scale_name =
        std::string("/") + ov::npuw::util::constants::past_key_values + "/value/scale";
    const std::string past_value_zp_name =
        std::string("/") + ov::npuw::util::constants::past_key_values + "/value/zp";
    const std::string past_key_label = std::string(ov::npuw::util::constants::past_key_values) + ".<N>.key";
    const std::string past_value_label = std::string(ov::npuw::util::constants::past_key_values) + ".<N>.value";

    bool found_key_cache_input = false;
    bool found_value_cache_input = false;
    bool found_key_scale_input = false;
    bool found_key_zp_input = false;
    bool found_value_scale_input = false;
    bool found_value_zp_input = false;

    for (const auto& input : model->inputs()) {
        if (ignore_quant_aux_ports &&
            (any_name_contains(input, "/scale") || any_name_contains(input, "/zp"))) {
            continue;
        }

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
                << past_key_label << " input must have type " << expected;
        }

        if (is_past_value) {
            found_value_cache_input = true;
            const auto expected = is_quantized ? precision_value_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(input.get_element_type(), expected)
                << past_value_label << " input must have type " << expected;
        }

        if (any_name_contains(input, past_key_scale_name)) {
            found_key_scale_input = true;
            const auto expected = precision_key_matrix().at(kv_type).at("scale");
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_key scale input must have type " << expected;
        }

        if (any_name_contains(input, past_key_zp_name)) {
            found_key_zp_input = true;
            const auto expected = precision_key_matrix().at(kv_type).at("zero_point");
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_key zero-point input must have type " << expected;
        }

        if (any_name_contains(input, past_value_scale_name)) {
            found_value_scale_input = true;
            const auto expected = precision_value_matrix().at(kv_type).at("scale");
            EXPECT_EQ(input.get_element_type(), expected)
                << "past_value scale input must have type " << expected;
        }

        if (any_name_contains(input, past_value_zp_name)) {
            found_value_zp_input = true;
        }
    }

    EXPECT_TRUE(found_key_cache_input) << "No " << past_key_label << " input found in model";
    EXPECT_TRUE(found_value_cache_input) << "No " << past_value_label << " input found in model";

    if (ignore_quant_aux_ports) {
        return;
    }

    if (is_quantized) {
        EXPECT_TRUE(found_key_scale_input)
            << "Asymmetric quantized KV key-cache must expose scale input";
        EXPECT_TRUE(found_key_zp_input)
            << "Asymmetric quantized KV key-cache must expose zero-point input";
        EXPECT_TRUE(found_value_scale_input)
            << "Symmetric quantized KV value-cache must expose scale input";
        EXPECT_FALSE(found_value_zp_input)
            << "Symmetric quantized KV value-cache must not expose zero-point input";
    } else {
        EXPECT_FALSE(found_key_scale_input) << "Non-quantized KV-cache must not expose key scale input";
        EXPECT_FALSE(found_key_zp_input) << "Non-quantized KV-cache must not expose key zero-point input";
        EXPECT_FALSE(found_value_scale_input) << "Non-quantized KV-cache must not expose value scale input";
        EXPECT_FALSE(found_value_zp_input) << "Non-quantized KV-cache must not expose value zero-point input";
    }
}

void expect_kv_cache_present_output_types(const std::shared_ptr<ov::Model>& model,
                                          const ov::element::Type kv_type,
                                          const bool ignore_quant_aux_ports = false) {
    const bool is_quantized = is_quantized_kv_type(kv_type);

    const std::string present_key_scale_name =
        std::string("/") + ov::npuw::util::constants::present + "/key/scale";
    const std::string present_key_zp_name =
        std::string("/") + ov::npuw::util::constants::present + "/key/zp";
    const std::string present_value_scale_name =
        std::string("/") + ov::npuw::util::constants::present + "/value/scale";
    const std::string present_value_zp_name =
        std::string("/") + ov::npuw::util::constants::present + "/value/zp";
    const std::string present_key_label = std::string(ov::npuw::util::constants::present) + ".<N>.key";
    const std::string present_value_label = std::string(ov::npuw::util::constants::present) + ".<N>.value";

    bool found_present_key = false;
    bool found_present_value = false;
    bool found_present_key_scale = false;
    bool found_present_key_zp = false;
    bool found_present_value_scale = false;
    bool found_present_value_zp = false;

    for (const auto& output : model->outputs()) {
        if (ignore_quant_aux_ports &&
            (any_name_contains(output, "/scale") || any_name_contains(output, "/zp"))) {
            continue;
        }

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
                << present_key_label << " output must have type " << expected;
        }

        if (is_present_value) {
            found_present_value = true;
            const auto expected = is_quantized ? precision_value_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(output.get_element_type(), expected)
                << present_value_label << " output must have type " << expected;
        }

        if (any_name_contains(output, present_key_scale_name)) {
            found_present_key_scale = true;
            const auto expected = precision_key_matrix().at(kv_type).at("scale");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present key scale output must have type " << expected;
        }

        if (any_name_contains(output, present_key_zp_name)) {
            found_present_key_zp = true;
            const auto expected = precision_key_matrix().at(kv_type).at("zero_point");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present key zero-point output must have type " << expected;
        }

        if (any_name_contains(output, present_value_scale_name)) {
            found_present_value_scale = true;
            const auto expected = precision_value_matrix().at(kv_type).at("scale");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present value scale output must have type " << expected;
        }

        if (any_name_contains(output, present_value_zp_name)) {
            found_present_value_zp = true;
        }
    }

    EXPECT_TRUE(found_present_key) << "No " << present_key_label << " output found in model";
    EXPECT_TRUE(found_present_value) << "No " << present_value_label << " output found in model";

    if (ignore_quant_aux_ports) {
        return;
    }

    if (is_quantized) {
        EXPECT_TRUE(found_present_key_scale)
            << "Asymmetric quantized KV key-cache must expose present scale output";
        EXPECT_TRUE(found_present_key_zp)
            << "Asymmetric quantized KV key-cache must expose present zero-point output";
        EXPECT_TRUE(found_present_value_scale)
            << "Symmetric quantized KV value-cache must expose present scale output";
        EXPECT_FALSE(found_present_value_zp)
            << "Symmetric quantized KV value-cache must not expose present zero-point output";
    } else {
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

// update_kvcache/copy_kvcache map output names to past-key input names. This must
// also work for quantized aux outputs (scale/zero-point) when graph rewrites add
// prefixes (e.g. DynamicDequantize/.../present...).
TEST_P(ConvertKVCacheHintPrecisionTest, GenerateModelKvOutputsResolveToPastInputsForKvUpdate) {
    const auto kv_type = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(make_kv_precision_props(kv_type), recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    std::unordered_set<std::string> input_names;
    for (const auto& input : generate.model->inputs()) {
        input_names.insert(input.get_any_name());
    }

    bool checked_any_kv_output = false;
    for (const auto& output : generate.model->outputs()) {
        const auto& output_name = output.get_any_name();
        const bool is_kv_output = output_name.find(ov::npuw::util::constants::present) != std::string::npos ||
                                  output_name.find(ov::npuw::util::constants::past_key_values) != std::string::npos;
        if (!is_kv_output) {
            continue;
        }

        checked_any_kv_output = true;
        const auto resolved_input = resolve_kv_input_name_for_test(output_name, input_names);
        ASSERT_TRUE(resolved_input.has_value())
            << "No matching past-key input for KV output name used by update flow: " << output_name;
    }
    ASSERT_TRUE(checked_any_kv_output) << "No KV-related outputs found in generate model";
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

    expect_kv_cache_input_types(model, kv_type, true);
}

TEST_P(ConvertKVCacheHintPrecisionTest, WhisperKVCacheModelPresentOutputsHaveExpectedPrecision) {
    const auto kv_type = GetParam();
    auto model = ov::test::npuw::build_whisper_decoder_test_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();
    ASSERT_TRUE(ov::npuw::util::PrepareWhisperKVCacheModel().run_on_model(model));
    ASSERT_TRUE(ov::npuw::ConvertKVCacheToPrecision(kv_type).run_on_model(model));

    expect_kv_cache_present_output_types(model, kv_type, true);
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

TEST_F(ConvertKVCacheToPrecisionPassTest, CopyKvCacheSimpleSmoke) {
    RecordingFactory recorder;
    auto compiled_unique = create_compiled_model(make_kv_precision_props(ov::element::i8), recorder);
    ASSERT_NE(compiled_unique, nullptr);

    std::shared_ptr<ov::npuw::LLMCompiledModel> compiled(compiled_unique.release());
    TestableLLMInferRequest request(compiled);
    request.prepare_non_chunked_copy();

    ASSERT_NO_THROW(request.copy_kvcache());
}

// Regression for kv-cache runtime copy path: execute real copy_kvcache() and verify
// that all KV outputs (including quantized aux tensors) are copied to matching past inputs.
TEST_F(ConvertKVCacheToPrecisionPassTest, CopyKvCacheCopiesQuantizedAuxTensorsByNameMapping) {
    RecordingFactory recorder;
    auto compiled_unique = create_compiled_model(make_kv_precision_props(ov::element::i8), recorder);
    ASSERT_NE(compiled_unique, nullptr);
    std::shared_ptr<ov::npuw::LLMCompiledModel> compiled(compiled_unique.release());
    TestableLLMInferRequest request(compiled);
    request.prepare_non_chunked_copy();

    std::unordered_set<std::string> input_names;
    for (const auto& [name, _] : request.kvcache_in_ports()) {
        input_names.insert(name);
    }
    struct CopyPair {
        std::string output_name;
        std::string input_name;
    };
    std::vector<CopyPair> copied_pairs;

    uint8_t pattern_seed = 7u;
    for (const auto& [output_name, _] : request.kvcache_out_ports()) {
        if (!is_kv_name(output_name)) {
            continue;
        }
        const auto resolved_input = resolve_kv_input_name_for_test(output_name, input_names);
        ASSERT_TRUE(resolved_input.has_value())
            << "No past KV input mapped for output: " << output_name;

        const auto& prefill_out_port = request.prefill_out_ports().at(output_name);
        auto src_tensor = request.prefill_request()->get_tensor(prefill_out_port);
        auto dst_tensor = request.kvcache_request()->get_tensor(request.kvcache_in_ports().at(resolved_input.value()));

        if (src_tensor->get_byte_size() == 0 || dst_tensor->get_byte_size() == 0) {
            continue;
        }

        ov::Tensor src_host(src_tensor->get_element_type(), src_tensor->get_shape());
        ov::Tensor dst_host(dst_tensor->get_element_type(), dst_tensor->get_shape());
        std::memset(src_host.data(), pattern_seed, src_host.get_byte_size());
        std::memset(dst_host.data(), static_cast<int>(pattern_seed + 1), dst_host.get_byte_size());
        ov::get_tensor_impl(src_host)->copy_to(src_tensor._ptr);
        ov::get_tensor_impl(dst_host)->copy_to(dst_tensor._ptr);
        pattern_seed = static_cast<uint8_t>(pattern_seed + 13);
        copied_pairs.push_back({output_name, resolved_input.value()});
    }

    ASSERT_FALSE(copied_pairs.empty()) << "No KV output/input pairs found for copy_kvcache test";

    ASSERT_NO_THROW(request.copy_kvcache());

    for (const auto& pair : copied_pairs) {
        auto src_tensor = request.prefill_request()->get_tensor(request.prefill_out_ports().at(pair.output_name));
        auto dst_tensor = request.kvcache_request()->get_tensor(request.kvcache_in_ports().at(pair.input_name));
        auto [src_view, dst_view] = make_non_chunked_copy_views(request, pair.output_name, src_tensor, dst_tensor);

        ASSERT_EQ(src_view->get_byte_size(), dst_view->get_byte_size())
            << "Byte-size mismatch for output/input pair: " << pair.output_name << " -> " << pair.input_name;

        ov::Tensor src_host(src_view->get_element_type(), src_view->get_shape());
        ov::Tensor dst_host(dst_view->get_element_type(), dst_view->get_shape());
        src_view->copy_to(ov::get_tensor_impl(src_host)._ptr);
        dst_view->copy_to(ov::get_tensor_impl(dst_host)._ptr);

        EXPECT_EQ(std::memcmp(src_host.data(), dst_host.data(), src_host.get_byte_size()), 0)
            << "copy_kvcache did not copy bytes for pair: " << pair.output_name << " -> " << pair.input_name;
    }
}

}  // namespace
