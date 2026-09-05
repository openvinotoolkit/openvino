// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/convert_kvcache_to_precision.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <cstring>
#include <regex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../util.hpp"
#include "infer_request_utils.hpp"
#include "llm_test_helpers.hpp"
#include "llm_infer_request.hpp"
#include "llm_pass_test_fixture.hpp"
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

class TestableLLMCompiledModel : public ov::npuw::LLMCompiledModel {
public:
    using ov::npuw::LLMCompiledModel::LLMCompiledModel;

    ov::npuw::LLMCompiledModel::KVCacheDesc& kvcache_desc() {
        return m_kvcache_desc;
    }

    const ov::npuw::LLMCompiledModel::KVCacheDesc& kvcache_desc() const {
        return m_kvcache_desc;
    }

    bool use_chunk_prefill() const {
        return m_use_chunk_prefill;
    }

    uint64_t prefill_chunk_size() const {
        return m_prefill_chunk_size;
    }
};

class TestableLLMInferRequest final : public ov::npuw::LLMInferRequest {
public:
    explicit TestableLLMInferRequest(const std::shared_ptr<TestableLLMCompiledModel>& compiled_model)
        : ov::npuw::LLMInferRequest(compiled_model), m_testable_model(compiled_model) {}

    using ov::npuw::LLMInferRequest::copy_kvcache;
    using ov::npuw::LLMInferRequest::update_kvcache_for;
    using ov::npuw::LLMInferRequest::clear_chunk_prefill_kv_cache;

    const std::vector<std::string>& kvcache_past_names() const {
        return m_kvcache_past_names;
    }

    void prepare_non_chunked_copy() {
        auto& desc = m_testable_model->kvcache_desc();
        ASSERT_FALSE(m_testable_model->use_chunk_prefill());
        ASSERT_GT(desc.max_prompt_size, 0u);
        desc.num_stored_tokens = desc.max_prompt_size;
    }

    std::pair<ov::SoPtr<ov::ITensor>, ov::SoPtr<ov::ITensor>> make_non_chunked_copy_views(
        const std::string& output_name,
        const ov::SoPtr<ov::ITensor>& src_tensor,
        const ov::SoPtr<ov::ITensor>& dst_tensor) const {
        const auto& desc = m_testable_model->kvcache_desc();
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

    void prepare_chunked_copy(uint32_t num_stored_tokens, uint64_t tokens_in_present_chunk) {
        auto& desc = m_testable_model->kvcache_desc();
        ASSERT_TRUE(m_testable_model->use_chunk_prefill());
        ASSERT_GT(m_testable_model->prefill_chunk_size(), 0u);
        ASSERT_GT(desc.max_prompt_size, 0u);
        desc.num_stored_tokens = num_stored_tokens;
        m_tokens_in_present_chunk = tokens_in_present_chunk;
    }

    uint64_t prefill_chunk_size() const {
        return m_testable_model->prefill_chunk_size();
    }

    bool past_kv_bound() const {
        return m_past_kv_bound;
    }

    // The unit-test harness has no NPU core, so the temp buffer allocated by the chunked
    // copy_kvcache() backup path (m_past_kv_bound) must live on CPU.
    void use_cpu_pre_alloc() {
        m_pre_alloc_device = "CPU";
    }

    // Returns {pre_kv_dim, gen_kv_dim} used by copy_kvcache() for the given output name.
    std::pair<uint32_t, uint32_t> kv_dims(const std::string& output_name) const {
        const auto& desc = m_testable_model->kvcache_desc();
        const auto is_value_tensor = output_name.find("value") != std::string::npos;
        const auto is_quant_aux_tensor = output_name.find("/scale") != std::string::npos ||
                                         output_name.find("/zp") != std::string::npos;
        if (is_value_tensor && is_quant_aux_tensor) {
            // Mirror the runtime shape-based detection: embedding is collapsed to 1 by DQ.
            const auto input_name =
                std::regex_replace(output_name, std::regex("present"), "past_key_values");
            auto detect = [&](bool v_trans) -> uint32_t {
                if (m_kvcache_in_ports.count(input_name)) {
                    const auto& sh =
                        m_kvcache_request->get_tensor(m_kvcache_in_ports.at(input_name))->get_shape();
                    if (sh.size() == 4u && sh[3] == 1u && sh[2] > 1u) return 2u;
                    if (sh.size() == 4u && sh[2] == 1u && sh[3] > 1u) return 3u;
                }
                return v_trans ? 3u : desc.dim;
            };
            return {detect(desc.v_tensors_transposed_pre), detect(desc.v_tensors_transposed_gen)};
        }
        const auto kv_dim = [&](bool v_transposed) {
            return (is_value_tensor && v_transposed) ? 3u : desc.dim;
        };
        return {kv_dim(desc.v_tensors_transposed_pre), kv_dim(desc.v_tensors_transposed_gen)};
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

private:
    std::shared_ptr<TestableLLMCompiledModel> m_testable_model;
};

bool is_kv_name(std::string_view name) {
    return ov::npuw::util::isKVCacheName(std::string(name));
}

ov::Tensor slice_to_host(const ov::SoPtr<ov::ITensor>& view) {
    ov::Tensor host(view->get_element_type(), view->get_shape());
    view->copy_to(ov::get_tensor_impl(host)._ptr);
    return host;
}

bool is_aux_kv_name(const std::string& name) {
    return name.find("scale") != std::string::npos || name.find("zp") != std::string::npos;
}

const std::map<ov::element::Type, std::map<std::string, ov::element::Type>>& precision_key_input_matrix() {
    static const std::map<ov::element::Type, std::map<std::string, ov::element::Type>> matrix = {
        {ov::element::u8, {{"value", ov::element::u8}, {"scale", ov::element::f32}, {"zero_point", ov::element::u8}}},
        {ov::element::i8, {{"value", ov::element::i8}, {"scale", ov::element::f32}, {"zero_point", ov::element::i8}}}};
    return matrix;
}

const std::map<ov::element::Type, std::map<std::string, ov::element::Type>>& precision_key_output_matrix() {
    static const std::map<ov::element::Type, std::map<std::string, ov::element::Type>> matrix = {
        {ov::element::u8, {{"value", ov::element::u8}, {"scale", ov::element::f32}, {"zero_point", ov::element::u8}}},
        {ov::element::i8, {{"value", ov::element::i8}, {"scale", ov::element::f32}, {"zero_point", ov::element::i8}}}};
    return matrix;
}

const std::map<ov::element::Type, std::map<std::string, ov::element::Type>>& precision_value_matrix() {
    static const std::map<ov::element::Type, std::map<std::string, ov::element::Type>> matrix = {
        {ov::element::u8, {{"value", ov::element::i8}, {"scale", ov::element::f32}}},
        {ov::element::i8, {{"value", ov::element::i8}, {"scale", ov::element::f32}}}};
    return matrix;
}

bool is_quantized_kv_type(const ov::element::Type kv_type) {
    return precision_key_input_matrix().count(kv_type) > 0;
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
            const auto expected = is_quantized ? precision_key_input_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(input.get_element_type(), expected)
                << past_key_label << " input must have type " << expected;
        }

        if (is_past_value) {
            found_value_cache_input = true;
            const auto expected = is_quantized ? precision_value_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(input.get_element_type(), expected)
                << past_value_label << " input must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(input, past_key_scale_name)) {
            found_key_scale_input = true;
            const auto expected = precision_key_input_matrix().at(kv_type).at("scale");
            EXPECT_EQ(input.get_element_type(), expected) << "past_key scale input must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(input, past_key_zp_name)) {
            found_key_zp_input = true;
            const auto expected = precision_key_input_matrix().at(kv_type).at("zero_point");
            EXPECT_EQ(input.get_element_type(), expected) << "past_key zero-point input must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(input, past_value_scale_name)) {
            found_value_scale_input = true;
            const auto expected = precision_value_matrix().at(kv_type).at("scale");
            EXPECT_EQ(input.get_element_type(), expected) << "past_value scale input must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(input, past_value_zp_name)) {
            found_value_zp_input = true;
        }
    }

    EXPECT_TRUE(found_key_cache_input) << "No " << past_key_label << " input found in model";
    EXPECT_TRUE(found_value_cache_input) << "No " << past_value_label << " input found in model";

    if (ignore_quant_aux_ports) {
        return;
    }

    if (is_quantized) {
        EXPECT_TRUE(found_key_scale_input) << "Asymmetric quantized KV key-cache must expose scale input";
        EXPECT_TRUE(found_key_zp_input) << "Asymmetric quantized KV key-cache must expose zero-point input";
        EXPECT_TRUE(found_value_scale_input) << "Symmetric quantized KV value-cache must expose scale input";
        EXPECT_FALSE(found_value_zp_input) << "Symmetric quantized KV value-cache must not expose zero-point input";
    } else if (!is_quantized) {
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
            const auto expected = is_quantized ? precision_key_input_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(output.get_element_type(), expected)
                << present_key_label << " output must have type " << expected;
        }

        if (is_present_value) {
            found_present_value = true;
            const auto expected = is_quantized ? precision_value_matrix().at(kv_type).at("value") : kv_type;
            EXPECT_EQ(output.get_element_type(), expected)
                << present_value_label << " output must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(output, present_key_scale_name)) {
            found_present_key_scale = true;
            const auto expected = precision_key_output_matrix().at(kv_type).at("scale");
            EXPECT_EQ(output.get_element_type(), expected) << "present key scale output must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(output, present_key_zp_name)) {
            found_present_key_zp = true;
            const auto expected = precision_key_output_matrix().at(kv_type).at("zero_point");
            EXPECT_EQ(output.get_element_type(), expected)
                << "present key zero-point output must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(output, present_value_scale_name)) {
            found_present_value_scale = true;
            const auto expected = precision_value_matrix().at(kv_type).at("scale");
            EXPECT_EQ(output.get_element_type(), expected) << "present value scale output must have type " << expected;
        }

        if (!ignore_quant_aux_ports && any_name_contains(output, present_value_zp_name)) {
            found_present_value_zp = true;
        }
    }

    EXPECT_TRUE(found_present_key) << "No " << present_key_label << " output found in model";
    EXPECT_TRUE(found_present_value) << "No " << present_value_label << " output found in model";

    if (ignore_quant_aux_ports) {
        return;
    }

    if (is_quantized) {
        EXPECT_TRUE(found_present_key_scale) << "Asymmetric quantized KV key-cache must expose present scale output";
        EXPECT_TRUE(found_present_key_zp) << "Asymmetric quantized KV key-cache must expose present zero-point output";
        EXPECT_TRUE(found_present_value_scale) << "Symmetric quantized KV value-cache must expose present scale output";
        EXPECT_FALSE(found_present_value_zp)
            << "Symmetric quantized KV value-cache must not expose present zero-point output";
    } else if (!is_quantized) {
        EXPECT_FALSE(found_present_key_scale) << "Non-quantized KV-cache must not expose present key scale output";
        EXPECT_FALSE(found_present_key_zp) << "Non-quantized KV-cache must not expose present key zero-point output";
        EXPECT_FALSE(found_present_value_scale) << "Non-quantized KV-cache must not expose present value scale output";
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
    ::testing::Values(ov::element::f16, ov::element::f8e4m3, ov::element::f8e5m2, ov::element::i8, ov::element::u8),
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
        const bool is_kv_output = ov::npuw::util::isKVCacheName(output_name);
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

class ConvertKVCacheToPrecisionPassTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    std::shared_ptr<TestableLLMCompiledModel> create_testable_model(const ov::AnyMap& extra_props,
                                                                    RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        std::unique_ptr<TestableLLMCompiledModel> model(new TestableLLMCompiledModel(
            ov::test::npuw::build_llm_test_model(), m_plugin, props, recorder.make_factory()));
        return std::shared_ptr<TestableLLMCompiledModel>(model.release());
    }
};

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

    ASSERT_NO_THROW(
        compiled = create_compiled_model({{"NPUW_LLM_PREFILL_HINT", "DYNAMIC"}, {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}},
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
    auto compiled = create_testable_model(make_kv_precision_props(ov::element::i8), recorder);
    ASSERT_NE(compiled, nullptr);
    ASSERT_GT(compiled->kvcache_desc().max_prompt_size, 0u) << "model kvcache_desc not initialized after construction";
    // Isolate: only test construction. If this passes, the crash is in LLMInferRequest ctor
    // TestableLLMInferRequest request(compiled);
    // request.prepare_non_chunked_copy();
    // ASSERT_NO_THROW(request.copy_kvcache());
}

// Regression for kv-cache runtime copy path: execute real copy_kvcache() and verify
// that all KV outputs (including quantized aux tensors) are copied to matching past inputs.
TEST_F(ConvertKVCacheToPrecisionPassTest, CopyKvCacheCopiesQuantizedAuxTensorsByNameMapping) {
    RecordingFactory recorder;
    auto compiled = create_testable_model(make_kv_precision_props(ov::element::i8), recorder);
    ASSERT_NE(compiled, nullptr);
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
        auto [src_view, dst_view] = request.make_non_chunked_copy_views(pair.output_name, src_tensor, dst_tensor);

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

TEST_F(ConvertKVCacheToPrecisionPassTest, CopyKvCacheUsesSequenceDimensionForValueAuxRoi) {
    RecordingFactory recorder;
    auto compiled = create_testable_model(make_kv_precision_props(ov::element::i8), recorder);
    ASSERT_NE(compiled, nullptr);
    TestableLLMInferRequest request(compiled);
    request.prepare_non_chunked_copy();

    bool found_value_aux = false;
    for (const auto& [output_name, output_port] : request.prefill_out_ports()) {
        if (output_name.find("/present/value/scale") == std::string::npos &&
            output_name.find("/present/value/zp") == std::string::npos) {
            continue;
        }

        const auto& tensor = request.prefill_request()->get_tensor(output_port);
        ASSERT_EQ(tensor->get_shape().size(), 4u);
        ASSERT_EQ(tensor->get_shape()[2], 1u);
        ASSERT_GE(tensor->get_shape()[3], compiled->kvcache_desc().max_prompt_size);

        const auto [pre_kv_dim, gen_kv_dim] = request.kv_dims(output_name);
        EXPECT_EQ(pre_kv_dim, 3u);
        EXPECT_EQ(gen_kv_dim, 3u);
        found_value_aux = true;
        break;
    }

    ASSERT_TRUE(found_value_aux) << "No quantized value scale/zp output with [B,H,1,S] layout found";
    ASSERT_NO_THROW(request.copy_kvcache());
}

// Chunked-prefill counterpart of the test above. Exercises the two-part chunked copy_kvcache()
// path (past chunks + last present chunk, incl. the m_past_kv_bound backup) for quantized aux
// tensors (scale/zero-point), verifying that every KV output is mapped to its past input by name.
TEST_F(ConvertKVCacheToPrecisionPassTest, ChunkedCopyKvCacheCopiesQuantizedAuxTensorsByNameMapping) {
    RecordingFactory recorder;
    ov::AnyMap props = make_kv_precision_props(ov::element::i8);
    props["NPUW_LLM_PREFILL_HINT"] = "DYNAMIC";
    props["NPUW_LLM_PREFILL_CHUNK_SIZE"] = "32";
    auto compiled = create_testable_model(props, recorder);
    ASSERT_NE(compiled, nullptr);
    ASSERT_TRUE(compiled->use_chunk_prefill());

    TestableLLMInferRequest request(compiled);
    // No NPU core in the unit-test harness; keep the chunked backup buffer on CPU.
    request.use_cpu_pre_alloc();

    const auto chunk_size = static_cast<uint32_t>(request.prefill_chunk_size());
    ASSERT_GT(chunk_size, 0u);
    const auto max_prompt = compiled->kvcache_desc().max_prompt_size;
    ASSERT_GT(max_prompt, 1u);

    // Model the tail of a chunked prefill: the last chunk holds a single token, and at least
    // one full prior chunk already sits in the past buffer (tokens_in_past > 0 exercises part 1).
    const uint32_t tokens_in_present = 1u;
    const uint32_t tokens_in_past = std::min(chunk_size, max_prompt - tokens_in_present);
    ASSERT_GT(tokens_in_past, 0u);
    const uint32_t num_stored = tokens_in_past + tokens_in_present;
    request.prepare_chunked_copy(num_stored, tokens_in_present);

    std::unordered_set<std::string> input_names;
    for (const auto& [name, _] : request.kvcache_in_ports()) {
        input_names.insert(name);
    }

    auto to_host = [](const ov::SoPtr<ov::ITensor>& view) {
        ov::Tensor host(view->get_element_type(), view->get_shape());
        view->copy_to(ov::get_tensor_impl(host)._ptr);
        return host;
    };

    struct ChunkPair {
        std::string output_name;
        std::string input_name;
        ov::Tensor expected_past;     // prefill past [0, tokens_in_past)  -> gen past head
        ov::Tensor expected_present;  // prefill present [C - M, C)        -> gen past tail
    };
    std::vector<ChunkPair> pairs;

    uint8_t pattern_seed = 11u;
    for (const auto& [output_name, _] : request.kvcache_out_ports()) {
        if (!is_kv_name(output_name)) {
            continue;
        }
        const auto resolved_input = resolve_kv_input_name_for_test(output_name, input_names);
        ASSERT_TRUE(resolved_input.has_value()) << "No past KV input mapped for output: " << output_name;
        const auto& input_name = resolved_input.value();
        if (request.prefill_in_ports().find(input_name) == request.prefill_in_ports().end()) {
            continue;
        }

        auto present_tensor = request.prefill_request()->get_tensor(request.prefill_out_ports().at(output_name));
        auto gen_past_tensor = request.kvcache_request()->get_tensor(request.kvcache_in_ports().at(input_name));
        auto prefill_past_tensor = request.prefill_request()->get_tensor(request.prefill_in_ports().at(input_name));
        if (present_tensor->get_byte_size() == 0 || gen_past_tensor->get_byte_size() == 0) {
            continue;
        }

        const auto [pre_kv_dim, gen_kv_dim] = request.kv_dims(output_name);

        // Distinct patterns: A fills gen past (and, when m_past_kv_bound, the aliased prefill past),
        // B fills prefill present, so the part-2 assertion below can't pass by coincidence.
        ov::Tensor gen_fill(gen_past_tensor->get_element_type(), gen_past_tensor->get_shape());
        std::memset(gen_fill.data(), pattern_seed, gen_fill.get_byte_size());
        ov::get_tensor_impl(gen_fill)->copy_to(gen_past_tensor._ptr);

        ov::Tensor present_fill(present_tensor->get_element_type(), present_tensor->get_shape());
        std::memset(present_fill.data(), static_cast<int>(pattern_seed + 5), present_fill.get_byte_size());
        ov::get_tensor_impl(present_fill)->copy_to(present_tensor._ptr);
        pattern_seed = static_cast<uint8_t>(pattern_seed + 13);

        auto past_src = ov::npuw::util::make_tensor_slice(prefill_past_tensor, pre_kv_dim, 0u, tokens_in_past);
        auto present_src =
            ov::npuw::util::make_tensor_slice(present_tensor, pre_kv_dim, chunk_size - tokens_in_present, chunk_size);
        pairs.push_back({output_name, input_name, to_host(past_src), to_host(present_src)});
    }

    ASSERT_FALSE(pairs.empty()) << "No KV output/input pairs found for chunked copy_kvcache test";

    ASSERT_NO_THROW(request.copy_kvcache());

    for (const auto& pair : pairs) {
        auto gen_past_tensor = request.kvcache_request()->get_tensor(request.kvcache_in_ports().at(pair.input_name));
        const auto gen_kv_dim = request.kv_dims(pair.output_name).second;
        auto past_dst = ov::npuw::util::make_tensor_slice(gen_past_tensor, gen_kv_dim, 0u, tokens_in_past);
        auto present_dst =
            ov::npuw::util::make_tensor_slice(gen_past_tensor, gen_kv_dim, tokens_in_past, num_stored);

        ov::Tensor past_host(past_dst->get_element_type(), past_dst->get_shape());
        ov::Tensor present_host(present_dst->get_element_type(), present_dst->get_shape());
        past_dst->copy_to(ov::get_tensor_impl(past_host)._ptr);
        present_dst->copy_to(ov::get_tensor_impl(present_host)._ptr);

        ASSERT_EQ(past_host.get_byte_size(), pair.expected_past.get_byte_size())
            << "Past-chunk byte-size mismatch for pair: " << pair.output_name << " -> " << pair.input_name;
        ASSERT_EQ(present_host.get_byte_size(), pair.expected_present.get_byte_size())
            << "Present-chunk byte-size mismatch for pair: " << pair.output_name << " -> " << pair.input_name;

        EXPECT_EQ(std::memcmp(past_host.data(), pair.expected_past.data(), past_host.get_byte_size()), 0)
            << "chunked copy_kvcache corrupted past chunk for pair: " << pair.output_name << " -> " << pair.input_name;
        EXPECT_EQ(std::memcmp(present_host.data(), pair.expected_present.data(), present_host.get_byte_size()), 0)
            << "chunked copy_kvcache did not copy present chunk for pair: " << pair.output_name << " -> "
            << pair.input_name;
    }
}

// Per generate-step KV update: update_kvcache_for() appends the newly produced token's KV from the
// generate present outputs into the past inputs, including quantized aux (scale/zero-point) tensors.
TEST_F(ConvertKVCacheToPrecisionPassTest, UpdateKvCacheForCopiesQuantizedAuxTensorsByNameMapping) {
    RecordingFactory recorder;
    auto compiled = create_testable_model(make_kv_precision_props(ov::element::i8), recorder);
    ASSERT_NE(compiled, nullptr);
    TestableLLMInferRequest request(compiled);

    auto& desc = compiled->kvcache_desc();
    ASSERT_GT(desc.max_prompt_size, 0u);
    const uint32_t num_tokens = 1u;
    const uint32_t num_stored = desc.max_prompt_size;
    desc.num_stored_tokens = num_stored;

    struct UpdatePair {
        std::string output_name;
        std::string input_name;
        ov::Tensor expected_tail;  // generate present [src_seq - num_tokens, src_seq)
    };
    std::vector<UpdatePair> pairs;
    bool saw_aux = false;

    uint8_t seed = 23u;
    for (const auto& input_name : request.kvcache_past_names()) {
        const auto output_name = ov::npuw::util::past_key_values_to_present_name(input_name);
        if (request.kvcache_out_ports().find(output_name) == request.kvcache_out_ports().end() ||
            request.kvcache_in_ports().find(input_name) == request.kvcache_in_ports().end()) {
            continue;
        }
        auto present = request.kvcache_request()->get_tensor(request.kvcache_out_ports().at(output_name));
        auto past = request.kvcache_request()->get_tensor(request.kvcache_in_ports().at(input_name));
        if (present->get_byte_size() == 0 || past->get_byte_size() == 0) {
            continue;
        }
        saw_aux = saw_aux || is_aux_kv_name(input_name);

        ov::Tensor present_fill(present->get_element_type(), present->get_shape());
        std::memset(present_fill.data(), seed, present_fill.get_byte_size());
        ov::get_tensor_impl(present_fill)->copy_to(present._ptr);
        ov::Tensor past_fill(past->get_element_type(), past->get_shape());
        std::memset(past_fill.data(), static_cast<int>(seed + 7), past_fill.get_byte_size());
        ov::get_tensor_impl(past_fill)->copy_to(past._ptr);
        seed = static_cast<uint8_t>(seed + 13);

        const auto gen_kv_dim = request.kv_dims(output_name).second;
        const auto src_seq = static_cast<uint32_t>(present->get_shape()[gen_kv_dim]);
        auto present_tail = ov::npuw::util::make_tensor_slice(present, gen_kv_dim, src_seq - num_tokens, src_seq);
        pairs.push_back({output_name, input_name, slice_to_host(present_tail)});
    }
    ASSERT_FALSE(pairs.empty()) << "No KV output/input pairs found for update_kvcache_for test";
    ASSERT_TRUE(saw_aux) << "i8 KV cache must expose scale/zero-point past inputs";

    ASSERT_NO_THROW(request.update_kvcache_for(request.kvcache_request(),
                                               request.kvcache_in_ports(),
                                               request.kvcache_out_ports(),
                                               num_tokens,
                                               desc.v_tensors_transposed_gen));

    for (const auto& pair : pairs) {
        auto past = request.kvcache_request()->get_tensor(request.kvcache_in_ports().at(pair.input_name));
        const auto gen_kv_dim = request.kv_dims(pair.output_name).second;
        auto dst_tail = ov::npuw::util::make_tensor_slice(past, gen_kv_dim, num_stored - num_tokens, num_stored);
        auto actual = slice_to_host(dst_tail);
        ASSERT_EQ(actual.get_byte_size(), pair.expected_tail.get_byte_size()) << pair.input_name;
        EXPECT_EQ(std::memcmp(actual.data(), pair.expected_tail.data(), actual.get_byte_size()), 0)
            << "update_kvcache_for did not append new-token KV for: " << pair.output_name << " -> " << pair.input_name;
    }
}

// on_reset()/clear_chunk_prefill_kv_cache() zero-fill the prefill past KV inputs; the broadened
// past-name set means quantized aux (scale/zero-point) inputs must be cleared as well.
TEST_F(ConvertKVCacheToPrecisionPassTest, ClearChunkPrefillZeroFillsQuantizedAuxPastTensors) {
    RecordingFactory recorder;
    ov::AnyMap props = make_kv_precision_props(ov::element::i8);
    props["NPUW_LLM_PREFILL_HINT"] = "DYNAMIC";
    props["NPUW_LLM_PREFILL_CHUNK_SIZE"] = "32";
    auto compiled = create_testable_model(props, recorder);
    ASSERT_NE(compiled, nullptr);
    TestableLLMInferRequest request(compiled);

    std::vector<std::string> filled;
    bool saw_aux = false;
    for (const auto& input_name : request.kvcache_past_names()) {
        if (request.prefill_in_ports().find(input_name) == request.prefill_in_ports().end()) {
            continue;
        }
        auto past = request.prefill_request()->get_tensor(request.prefill_in_ports().at(input_name));
        if (past->get_byte_size() == 0) {
            continue;
        }
        saw_aux = saw_aux || is_aux_kv_name(input_name);
        ov::Tensor fill(past->get_element_type(), past->get_shape());
        std::memset(fill.data(), 0xAB, fill.get_byte_size());
        ov::get_tensor_impl(fill)->copy_to(past._ptr);
        filled.push_back(input_name);
    }
    ASSERT_FALSE(filled.empty()) << "No prefill past KV inputs found to clear";
    ASSERT_TRUE(saw_aux) << "i8 KV cache must expose scale/zero-point past inputs";

    ASSERT_NO_THROW(request.clear_chunk_prefill_kv_cache());

    for (const auto& input_name : filled) {
        auto past = request.prefill_request()->get_tensor(request.prefill_in_ports().at(input_name));
        auto host = slice_to_host(past);
        const auto* bytes = static_cast<const uint8_t*>(host.data());
        const bool all_zero = std::all_of(bytes, bytes + host.get_byte_size(), [](uint8_t b) {
            return b == 0u;
        });
        EXPECT_TRUE(all_zero) << "clear_chunk_prefill_kv_cache left non-zero bytes in: " << input_name;
    }
}

}  // namespace
