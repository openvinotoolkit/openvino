// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa_compiled_model.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <typeinfo>
#include <unordered_set>
#include <utility>

#include "infer_request_utils.hpp"
#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "npuw_transformations/collapse_unqdq.hpp"
#include "npuw_transformations/conv_to_matmul.hpp"
#include "npuw_transformations/drop_zp_subtract.hpp"
#include "npuw_transformations/reshape_to_static_gqa.hpp"
#include "npuw_transformations/untangle_dq_scale.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "util.hpp"

namespace {

// Reads an NPUW option from a raw properties AnyMap, falling back to its default.
template <typename T>
auto cfg_get(const ov::AnyMap& properties) -> typename T::ValueType {
    const auto& opt_name = std::string(T::key());
    if (properties.count(opt_name)) {
        return properties.at(opt_name).as<typename T::ValueType>();
    }
    return T::defaultValue();
}

// Enables "[GQA-TRACE]" diagnostic logging via OPENVINO_NPUW_GQA_LOG=1 (developer builds only).
bool gqa_trace_enabled() {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static const bool enabled = [] {
        const char* value = std::getenv("OPENVINO_NPUW_GQA_LOG");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
#else
    return false;
#endif
}

// Strictly parses env var `env_name` as a positive integer; throws on malformed values.
std::optional<size_t> parse_positive_size_env_var(const char* env_name) {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    const char* raw = std::getenv(env_name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    const std::string str(raw);
    auto parsed = ov::npuw::util::parse_positive_int_env_var(str);
    if (!parsed) {
        OPENVINO_THROW(env_name, " must be a positive integer, got: '", str, "'");
    }
    return static_cast<size_t>(*parsed);
#else
    (void)env_name;
    return std::nullopt;
#endif
}

// OPENVINO_NPUW_GQA_CTX_LEN env override for the static KV-cache reshape target; takes
// priority over NPUW_LLM_MAX_CONTEXT_LEN.
std::optional<size_t> gqa_ctx_len_env_override() {
    static const std::optional<size_t> value = parse_positive_size_env_var("OPENVINO_NPUW_GQA_CTX_LEN");
    return value;
}

// OPENVINO_NPUW_GQA_SLIDING_LEN env override: best-effort default sliding-window capacity
// when a layer has no rt_info/op-attribute signal. Lower priority than NPUW_LLM_SWA_LEN.
std::optional<size_t> gqa_default_sliding_window_len_env_override() {
    static const std::optional<size_t> value = parse_positive_size_env_var("OPENVINO_NPUW_GQA_SLIDING_LEN");
    return value;
}

// Resolves the sliding-window fallback capacity: NPUW_LLM_SWA_LEN property (if set), else
// the env override, else nullopt. Never overrides rt_info or the op's own attribute.
std::optional<size_t> resolve_default_sliding_window_capacity(const ov::AnyMap& properties) {
    const uint32_t configured = cfg_get<::intel_npu::NPUW_LLM_SWA_LEN>(properties);
    if (configured != 0) {
        return static_cast<size_t>(configured);
    }
    return gqa_default_sliding_window_len_env_override();
}

void merge_config_with(ov::AnyMap& lhs, const ov::AnyMap& rhs) {
    for (const auto& [key, value] : rhs) {
        if (auto it = lhs.find(key); it != lhs.end()) {
            it->second = value;
        } else {
            lhs.emplace(key, value);
        }
    }
}

enum class GQAModelStage {
    UNKNOWN,
    PREFILL,
    GENERATE,
};

// V2 models ship weights as 2-bit group-quantized (u2) Parameters, unlike V0/V1's u4/u8.
bool has_2bit_group_quantized_weights(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_element_type() == ov::element::u2) {
            return true;
        }
    }
    return false;
}

std::pair<ov::AnyMap, GQAModelStage> with_gqa_defaults(const std::shared_ptr<ov::Model>& model,
                                                       const ov::AnyMap& properties,
                                                       ov::npuw::GQACompiledModel::Case gqa_case) {
    const auto detect_gqa_model_stage = [&]() {
        // The activation tensor ("input_hidden_states") is what the embedding
        // model stage feeds into the transformer.  For the generate (iter) model
        // the seq dim is specialized to 1 via a free-dim override; for the
        // prefill (ctx) model the seq dim is either >1 or left dynamic (the
        // context length varies at runtime).  KV-cache slicing has no effect on
        // this input at all.
        for (const auto& parameter : model->get_parameters()) {
            const auto& name = parameter->get_friendly_name();
            if (name != "input_hidden_states" && name != "input_ids") {
                continue;
            }

            const auto& partial_shape = parameter->get_partial_shape();
            if (partial_shape.rank().is_dynamic() || partial_shape.rank().get_length() < 2) {
                continue;
            }

            const auto& token_dim = partial_shape[1];

            // Dynamic seq dim → prefill model with variable context length.
            if (token_dim.is_dynamic()) {
                return GQAModelStage::PREFILL;
            }

            const auto token_count = token_dim.get_length();
            if (token_count == 1) {
                return GQAModelStage::GENERATE;
            }
            return GQAModelStage::PREFILL;
        }
        return GQAModelStage::UNKNOWN;
    };

    const auto get_model_max_seq_length = [&]() {
        std::size_t max_seq_length = 0;
        for (const auto& res : model->get_parameters()) {
            if (res->get_friendly_name().find("past_keys_") == std::string::npos) {
                continue;
            }

            const auto& past_keys_shape = res->get_output_partial_shape(0);
            if (past_keys_shape.rank().is_static() && past_keys_shape.size() > 2) {
                const auto& seq_dim = past_keys_shape[2];
                if (seq_dim.is_static()) {
                    max_seq_length = static_cast<std::size_t>(seq_dim.get_length());
                    break;
                }
            }
            break;
        }

        return max_seq_length;
    };

    const auto has_transposed_value_tensors = [&]() {
        ov::PartialShape past_keys_shape;
        ov::PartialShape past_values_shape;

        for (const auto& parameter : model->get_parameters()) {
            const auto& name = parameter->get_friendly_name();
            if (name.find("past_keys_") != std::string::npos) {
                past_keys_shape = parameter->get_partial_shape();
            } else if (name.find("past_values_") != std::string::npos) {
                past_values_shape = parameter->get_partial_shape();
            }
        }

        if (past_keys_shape.rank().is_static() && past_values_shape.rank().is_static() && past_keys_shape.size() >= 4 &&
            past_values_shape.size() >= 4) {
            const auto& key_seq_dim = past_keys_shape[2];
            const auto& key_dim = past_keys_shape[3];
            const auto& value_seq_dim = past_values_shape[2];
            const auto& value_dim = past_values_shape[3];

            if (key_seq_dim.is_static() && key_dim.is_static() && value_seq_dim.is_static() && value_dim.is_static()) {
                return (value_seq_dim.get_length() == key_dim.get_length() &&
                        value_dim.get_length() == key_seq_dim.get_length());
            }
        }

        return false;
    };

    // V0/V1 are pre-assembled GQA graphs that online partitioning would only get in the way
    // of. V2 (2-bit group-quantized weights) still benefits from online partitioning: its
    // prefill stage switches to the REP (repeated-block) pipeline below, and both stages
    // compile their partitioned blocks in parallel via NPUW_PARALLEL_COMPILE; V0/V1 don't
    // use online partitioning at all, so that option would have no effect for them.
    const bool is_v2 = gqa_case == ov::npuw::GQACompiledModel::Case::V2;
    ov::AnyMap config = {{std::string(::intel_npu::COMPILER_DYNAMIC_QUANTIZATION::key()), "YES"},
                         {std::string(::intel_npu::NPUW_DEVICES::key()), "NPU"},
                         {ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SPEED},
                         {std::string(::intel_npu::NPUW_UNQDQ::key()), "YES"},
                         {std::string(::intel_npu::COMPILATION_MODE_PARAMS::key()), "optimization-level=3"}};
    if (!is_v2) {
        config.emplace(std::string(::intel_npu::NPUW_ONLINE_PIPELINE::key()), "NONE");
    } else {
        config.emplace(std::string(::intel_npu::NPUW_PARALLEL_COMPILE::key()), "YES");
    }

    const auto stage = detect_gqa_model_stage();
    if (stage == GQAModelStage::PREFILL) {
        merge_config_with(config,
                          {{std::string(::intel_npu::NPUW_FOLD::key()), "YES"},
                           {std::string(::intel_npu::NPUW_FOLD_ONLY::key()), "attn"},
                           {std::string(::intel_npu::NPUW_ONLINE_ISOLATE::key()), "ATTN"},
                           {std::string(::intel_npu::NPUW_ONLINE_KEEP_BLOCKS_TAGGED::key()), "attn"},
                           {std::string(::intel_npu::NPUW_ATTN::key()), "STATIC"}});
        if (is_v2) {
            merge_config_with(config, {{std::string(::intel_npu::NPUW_ONLINE_PIPELINE::key()), "REP"}});
        }
        LOG_INFO("Detected prefill-style GQA model; applying FOLD with ATTN isolation");
    } else if (stage == GQAModelStage::GENERATE) {
        merge_config_with(config,
                          {{std::string(::intel_npu::NPUW_FOLD::key()), "YES"},
                           {std::string(::intel_npu::NPUW_FOLD_ONLY::key()), "attn"},
                           {std::string(::intel_npu::NPUW_FUNCALL_ASYNC::key()), "YES"},
                           {std::string(::intel_npu::NPUW_UNFOLD_IREQS::key()), "YES"}});

        // WA: Disable NPUW_UNFOLD_IREQS with long sequence lengths and not transposed value tensors.
        const auto max_seq_length = get_model_max_seq_length();
        const auto transposed_value_tensors = has_transposed_value_tensors();
        if (max_seq_length > 8192 && !transposed_value_tensors) {
            merge_config_with(config, {{std::string(::intel_npu::NPUW_UNFOLD_IREQS::key()), "NO"}});
            LOG_INFO("Detected generate-style GQA model with max sequence length "
                     << max_seq_length << " and transposed value tensors=" << transposed_value_tensors
                     << "; disabling NPUW_UNFOLD_IREQS");
        }
        LOG_INFO("Detected generate-style GQA model; applying FOLD with async funcall");
    } else {
        LOG_INFO("GQA model stage unknown; FOLD disabled");
    }
    merge_config_with(config, properties);
    // LLM-only option; the inner CompiledModel only registers ROOT options.
    config.erase(std::string(::intel_npu::NPUW_LLM_MAX_CONTEXT_LEN::key()));
    config.erase(std::string(::intel_npu::NPUW_LLM_SWA_LEN::key()));
    return {config, stage};
}

}  // namespace

// Prints a "[GQA-TRACE]" line when OPENVINO_NPUW_GQA_LOG=1 is set; no-op otherwise.
// `msg` is a chain of `<<`-streamed expressions. Developer builds only.
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
#    define GQA_TRACE(msg)                                       \
        do {                                                     \
            if (gqa_trace_enabled()) {                           \
                std::cout << "[GQA-TRACE] " << msg << std::endl; \
            }                                                    \
        } while (0)
#else
#    define GQA_TRACE(msg)
#endif

ov::npuw::GQACompiledModel::PreparedState ov::npuw::GQACompiledModel::prepare(const std::shared_ptr<ov::Model>& model,
                                                                              const ov::AnyMap& properties) {
    auto [prepared_properties, stage] = with_gqa_defaults(model, properties, identify_case(model));

    static std::atomic<std::size_t> mcount{0u};

    model->set_friendly_name(model->get_friendly_name() + "_gqa_" +
                             (stage == GQAModelStage::PREFILL    ? "prefill"
                              : stage == GQAModelStage::GENERATE ? "generate"
                                                                 : "unknown") +
                             std::to_string(mcount++));

    // Untangle shared scale constants so every DequantizeLinear Multiply
    // gets its own copy.  Some exporters reuse a single scale node across
    // multiple layers; NPUW's FOLD pass requires per-instance scalars.
    ov::npuw::UntangleDQScale untangle_dq_scale;
    untangle_dq_scale.run_on_model(model);
    // Drop all-zero zero-point Subtract nodes so ConvToMatMul sees a clean
    // Convert(Parameter) → Multiply(scale) weight chain.
    ov::npuw::DropZPSubtract drop_zp_subtract;
    drop_zp_subtract.run_on_model(model);
    // Rewrite 1x1 Convolutions with compressed (Parameter-sourced) weights as
    // MatMul + scale Multiply, keeping the Parameter shapes intact.
    ov::npuw::ConvToMatMul conv_to_matmul;
    conv_to_matmul.run_on_model(model);
    // Collapse FakeQuantize-based QDQ chains when requested.
    if (prepared_properties.at(std::string(::intel_npu::NPUW_UNQDQ::key())).as<bool>()) {
        ov::npuw::CollapseUNQDQ collapse_unqdq;
        collapse_unqdq.run_on_model(model);
    }

    // Reshape kv-cache to static, if any
    std::shared_ptr<ov::Model> compiled_model = model;
    std::unordered_map<std::string, size_t> dynamic_kv_cache_axes;
    uint32_t effective_max_context_len = cfg_get<::intel_npu::NPUW_LLM_MAX_CONTEXT_LEN>(properties);
    uint32_t effective_sliding_window_capacity = 0;
    if (has_dynamic_max_seq_len(model)) {
        const size_t max_seq_len = gqa_ctx_len_env_override().value_or(effective_max_context_len);
        effective_max_context_len = static_cast<uint32_t>(max_seq_len);
        compiled_model = model->clone();
        const auto default_sliding_window_capacity = resolve_default_sliding_window_capacity(properties);
        effective_sliding_window_capacity = static_cast<uint32_t>(default_sliding_window_capacity.value_or(0));
        ov::npuw::ReshapeToStaticGQA reshape_to_static_gqa(max_seq_len, default_sliding_window_capacity);
        reshape_to_static_gqa.run_on_model(compiled_model);
        dynamic_kv_cache_axes = reshape_to_static_gqa.dynamic_kv_cache_axes();
    }

    return {model,
            compiled_model,
            std::move(prepared_properties),
            std::move(dynamic_kv_cache_axes),
            effective_max_context_len,
            effective_sliding_window_capacity};
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::GQACompiledModel::make_compiled_model(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    return std::make_shared<ov::npuw::CompiledModel>(model, plugin, properties);
}

namespace {

// Resolves the Parameter feeding the GQA op's TOTAL_SEQUENCE_LENGTH input, used to
// compute an accurate present_key/present_value length at infer() time.
std::optional<std::string> find_total_sequence_length_param_name(const std::shared_ptr<const ov::Model>& model) {
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    constexpr size_t total_seq_len_idx = static_cast<size_t>(GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH);
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa || gqa->get_input_size() <= total_seq_len_idx) {
            continue;
        }
        if (auto param =
                ov::npuw::util::trace_back_to_parameter(gqa->input_value(total_seq_len_idx).get_node_shared_ptr())) {
            return param->get_friendly_name();
        }
    }
    return std::nullopt;
}

// Detects present_key/present_value names (underscore and HF-ONNX dot style). Shared by
// identify_case() and find_dynamic_kv_cache_output_axes() so both agree on a match.
bool is_present_kv_cache_output_name(const std::string& name) {
    return ov::npuw::util::contains_ignore_case(name, "present_key") ||
           ov::npuw::util::contains_ignore_case(name, "present_value") ||
           ov::npuw::util::contains_ignore_case(name, "present.");
}

// True if `gqa` is well-formed enough to count as real GQA evidence. Shared by
// find_valid_gqa_op() and find_all_valid_gqa_ops().
bool is_valid_gqa_op(const ov::op::internal::GroupQueryAttention& gqa) {
    using ov::op::internal::GroupQueryAttentionInputs;
    constexpr size_t mandatory_inputs = static_cast<size_t>(GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH) + 1;

    const auto num_heads = gqa.get_num_heads();
    const auto kv_num_heads = gqa.get_kv_num_heads();
    if (num_heads <= 0 || kv_num_heads <= 0 || num_heads % kv_num_heads != 0 ||
        gqa.get_input_size() < mandatory_inputs) {
        return false;  // malformed instance, doesn't count as evidence
    }
    if (gqa.get_do_rotary() && gqa.get_input_size() <= static_cast<size_t>(GroupQueryAttentionInputs::SIN_CACHE)) {
        return false;  // rotary requires the cos/sin cache inputs
    }
    return true;
}

std::vector<std::shared_ptr<ov::op::internal::GroupQueryAttention>> find_all_valid_gqa_ops(
    const std::shared_ptr<const ov::Model>& model) {
    using ov::op::internal::GroupQueryAttention;
    std::vector<std::shared_ptr<GroupQueryAttention>> result;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (gqa && is_valid_gqa_op(*gqa)) {
            result.push_back(std::move(gqa));
        }
    }
    return result;
}

std::shared_ptr<ov::op::internal::GroupQueryAttention> find_valid_gqa_op(
    const std::shared_ptr<const ov::Model>& model) {
    using ov::op::internal::GroupQueryAttention;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (gqa && is_valid_gqa_op(*gqa)) {
            return gqa;
        }
    }
    return nullptr;
}

}  // namespace

bool ov::npuw::GQACompiledModel::supports(const std::shared_ptr<const ov::Model>& model) {
    // identify_case() classifies GQA family regardless of dynamic shapes; this wrapper is
    // only needed when max_seq_len is actually dynamic.
    return identify_case(model) != Case::Unknown && has_dynamic_max_seq_len(model);
}

ov::npuw::GQACompiledModel::Case ov::npuw::GQACompiledModel::identify_case(
    const std::shared_ptr<const ov::Model>& model) {
    auto gqa = find_valid_gqa_op(model);
    if (!gqa) {
        return Case::Unknown;
    }

    // Each matched name must also be forward-reachable to the GQA op (not just same-named).
    const auto* gqa_node = gqa.get();

    // V1 uses `position_ids` for RoPE; V0/V2 use `past_seq_len`/`total_seq_len` instead.
    // V2 is further distinguished by 2-bit group-quantized (u2) weights.
    bool has_activation_input = false;
    bool has_position_ids = false;
    bool has_seq_len_signal = false;
    bool has_past_kv_cache = false;
    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (!ov::npuw::util::reaches(parameter.get(), gqa_node)) {
            continue;
        }
        has_activation_input = has_activation_input || (name == "input_hidden_states" || name == "input_ids");
        has_position_ids = has_position_ids || (name == "position_ids");
        has_seq_len_signal = has_seq_len_signal || (name == "past_seq_len") || (name == "total_seq_len");
        has_past_kv_cache = has_past_kv_cache || util::contains_ignore_case(name, "past_key") ||
                            util::contains_ignore_case(name, "past_value");
    }
    if (!has_activation_input || !has_past_kv_cache || (!has_position_ids && !has_seq_len_signal)) {
        return Case::Unknown;
    }

    bool has_present_kv_cache = false;
    for (const auto& result : model->get_results()) {
        const auto& name = result->get_friendly_name();
        if (is_present_kv_cache_output_name(name) && ov::npuw::util::reaches(gqa_node, result.get())) {
            has_present_kv_cache = true;
            break;
        }
    }
    if (!has_present_kv_cache) {
        return Case::Unknown;
    }

    if (has_position_ids) {
        return Case::V1;
    }
    return has_2bit_group_quantized_weights(model) ? Case::V2 : Case::V0;
}

bool ov::npuw::GQACompiledModel::has_dynamic_max_seq_len(const std::shared_ptr<const ov::Model>& model) {
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa) {
            continue;
        }
        for (auto kv_input : {GroupQueryAttentionInputs::PAST_KEY, GroupQueryAttentionInputs::PAST_VALUE}) {
            const auto& shape = gqa->input_value(static_cast<size_t>(kv_input)).get_partial_shape();
            if (auto axis = util::find_single_dynamic_axis(shape);
                axis && util::is_supported_kv_cache_capacity_axis(*axis, shape.size())) {
                return true;
            }
        }
        // Attention bias shares the KV-cache's max_seq_len dim; capacity is the LAST axis.
        if (gqa->get_input_size() > static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)) {
            const auto& bias_shape =
                gqa->input_value(static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)).get_partial_shape();
            if (bias_shape.rank().is_static()) {
                if (auto axis = util::find_single_dynamic_axis(bias_shape); axis && *axis == bias_shape.size() - 1) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::unordered_map<std::string, size_t> ov::npuw::GQACompiledModel::find_dynamic_kv_cache_output_axes(
    const std::vector<ov::Output<const ov::Node>>& outer_outputs) {
    std::unordered_map<std::string, size_t> result;
    for (const auto& output : outer_outputs) {
        const auto& full_name = output.get_node()->get_friendly_name();
        // ONNX frontend appends "/sink_port_0" to Result names; match before the first '/'.
        const auto slash_pos = full_name.find('/');
        const auto bare_name = slash_pos == std::string::npos ? full_name : full_name.substr(0, slash_pos);
        if (!is_present_kv_cache_output_name(bare_name)) {
            continue;
        }
        auto axis = util::find_single_dynamic_axis(output.get_partial_shape());
        if (!axis || !util::is_supported_kv_cache_capacity_axis(*axis, output.get_partial_shape().size())) {
            // No resolvable capacity axis; get_tensor() will fall back.
            continue;
        }
        result.emplace(full_name, *axis);
    }
    return result;
}

std::unordered_map<std::string, std::string> ov::npuw::GQACompiledModel::find_present_to_past_param_names(
    const std::shared_ptr<const ov::Model>& model) {
    using ov::op::internal::GroupQueryAttentionInputs;
    std::unordered_map<std::string, std::string> result;
    for (const auto& gqa : find_all_valid_gqa_ops(model)) {
        // Output 1/2 are present_key/present_value, matching PAST_KEY/PAST_VALUE inputs.
        static constexpr std::pair<size_t, GroupQueryAttentionInputs> kPresentPastPairs[] = {
            {1, GroupQueryAttentionInputs::PAST_KEY},
            {2, GroupQueryAttentionInputs::PAST_VALUE},
        };
        for (const auto& [output_idx, past_input] : kPresentPastPairs) {
            const auto past_idx = static_cast<size_t>(past_input);
            if (gqa->get_output_size() <= output_idx || gqa->get_input_size() <= past_idx) {
                continue;
            }
            auto past_param = ov::npuw::util::trace_back_to_parameter(gqa->input_value(past_idx).get_node_shared_ptr());
            auto present_result = ov::npuw::util::trace_forward_to_result(gqa->output(output_idx));
            if (!past_param || !present_result) {
                continue;
            }
            result.emplace(present_result->get_friendly_name(), past_param->get_friendly_name());
        }
    }
    return result;
}

ov::npuw::GQACompiledModel::GQACompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::AnyMap& properties,
                                             CompiledModelFactory factory)
    : GQACompiledModel(prepare(model, properties), plugin, std::move(factory)) {}

ov::npuw::GQACompiledModel::GQACompiledModel(PreparedState prepared,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             CompiledModelFactory factory)
    : ov::npuw::ICompiledModel(prepared.model, plugin),
      m_inner_compiled_model(factory(prepared.compiled_model, plugin, prepared.properties)),
      m_outer_model(prepared.model),
      m_dynamic_kv_cache_axes(std::move(prepared.dynamic_kv_cache_axes)),
      m_effective_max_context_len(prepared.effective_max_context_len),
      m_effective_sliding_window_capacity(prepared.effective_sliding_window_capacity) {
    OPENVINO_ASSERT(m_inner_compiled_model != nullptr, "GQACompiledModel requires a valid inner compiled model");
    m_dynamic_kv_cache_output_axes = find_dynamic_kv_cache_output_axes(outputs());
    m_total_seq_len_param_name = find_total_sequence_length_param_name(m_outer_model);
    m_present_to_past_param_names = find_present_to_past_param_names(m_outer_model);
}

// Used by import_model() to wrap a deserialized inner compiled model with the rebuilt
// outer model, so a cache hit produces a fully functional GQACompiledModel.
ov::npuw::GQACompiledModel::GQACompiledModel(const std::shared_ptr<ov::Model>& outer_model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             std::shared_ptr<ov::npuw::ICompiledModel> inner_compiled_model,
                                             std::unordered_map<std::string, size_t> dynamic_kv_cache_axes,
                                             uint32_t effective_max_context_len,
                                             uint32_t effective_sliding_window_capacity,
                                             std::optional<std::string> total_seq_len_param_name,
                                             std::unordered_map<std::string, std::string> present_to_past_param_names)
    : ov::npuw::ICompiledModel(outer_model, plugin),
      m_inner_compiled_model(std::move(inner_compiled_model)),
      m_outer_model(outer_model),
      m_dynamic_kv_cache_axes(std::move(dynamic_kv_cache_axes)),
      m_effective_max_context_len(effective_max_context_len),
      m_effective_sliding_window_capacity(effective_sliding_window_capacity) {
    OPENVINO_ASSERT(m_inner_compiled_model != nullptr, "GQACompiledModel requires a valid inner compiled model");
    m_dynamic_kv_cache_output_axes = find_dynamic_kv_cache_output_axes(outputs());
    // NOT re-derived via find_total_sequence_length_param_name()/find_present_to_past_param_names()
    // here: outer_model on this (import) path is a dummy carcass with no real GQA ops, so
    // those graph-walking helpers would silently find nothing. Use what was serialized
    // from the original compile, where the real model was available.
    m_total_seq_len_param_name = std::move(total_seq_len_param_name);
    m_present_to_past_param_names = std::move(present_to_past_param_names);
}

// Hand-rolled port serialization instead of ov::pass::Serialize: the IR writer recomputes
// a Result's shape/type from its input node, which would collapse our dummy-placeholder
// carcass Results down to the placeholder's shape.
void ov::npuw::GQACompiledModel::write_port_list(std::ostream& stream,
                                                 const std::vector<ov::Output<const ov::Node>>& ports) {
    using namespace ov::npuw::s11n;
    write(stream, ports.size());
    for (const auto& port : ports) {
        write(stream, port.get_node()->get_friendly_name());
        write(stream, port.get_element_type().to_string());
        write(stream, port.get_partial_shape().to_string());
        // Preserve tensor names too: ORT/OVEP match ports by these, not friendly names.
        write(stream, port.get_names());
    }
}

ov::ParameterVector ov::npuw::GQACompiledModel::read_input_port_list(std::istream& stream) {
    using namespace ov::npuw::s11n;
    size_t count = 0;
    read(stream, count);
    // Guard against a corrupted blob surfacing as an opaque bad_alloc instead of a clear error.
    OPENVINO_ASSERT(count <= 1024, "GQACompiledModel: implausible outer input port count (", count, ") on import");
    ov::ParameterVector params;
    params.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        std::string name, elem_type_str, shape_str;
        std::unordered_set<std::string> tensor_names;
        read(stream, name);
        read(stream, elem_type_str);
        read(stream, shape_str);
        read(stream, tensor_names);
        auto param =
            std::make_shared<ov::op::v0::Parameter>(ov::element::Type(elem_type_str), ov::PartialShape(shape_str));
        param->set_friendly_name(name);
        param->output(0).get_tensor().set_names(tensor_names);
        params.push_back(param);
    }
    return params;
}

ov::NodeVector ov::npuw::GQACompiledModel::read_output_port_list(std::istream& stream) {
    using namespace ov::npuw::s11n;
    size_t count = 0;
    read(stream, count);
    OPENVINO_ASSERT(count <= 1024, "GQACompiledModel: implausible outer output port count (", count, ") on import");
    ov::NodeVector results;
    results.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        std::string name, elem_type_str, shape_str;
        std::unordered_set<std::string> tensor_names;
        read(stream, name);
        read(stream, elem_type_str);
        read(stream, shape_str);
        read(stream, tensor_names);
        const auto elem_type = ov::element::Type(elem_type_str);
        // A dummy Constant source, purely so Result has something to wrap -- its own
        // reported type/shape is overridden right below via a fresh descriptor::Tensor,
        // matching the pattern used by ov::npuw::orc::serialize(shared_ptr<Node>&).
        auto dummy_source = std::make_shared<ov::op::v0::Constant>(elem_type, ov::Shape{1});
        auto result = std::make_shared<ov::op::v0::Result>(dummy_source);
        result->output(0).set_tensor_ptr(
            std::make_shared<ov::descriptor::Tensor>(elem_type, ov::PartialShape(shape_str), tensor_names));
        result->set_friendly_name(name);
        results.push_back(result);
    }
    return results;
}

void ov::npuw::GQACompiledModel::export_model(std::ostream& stream) const {
    LOG_INFO("Exporting GQACompiledModel...");
    LOG_BLOCK();
    GQA_TRACE("GQACompiledModel::export_model() begin, m_dynamic_kv_cache_axes has " << m_dynamic_kv_cache_axes.size()
                                                                                     << " entries:");
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    for (const auto& [name, axis] : m_dynamic_kv_cache_axes) {
        GQA_TRACE("    '" << name << "' -> axis " << axis);
    }
#endif
    using namespace ov::npuw::s11n;
    write_header(stream, NPUW_GQA_COMPILED_MODEL_INDICATOR);

    // Preserve the outer-facing ports and axis map so import_model() can rebuild an
    // equivalent GQACompiledModel, not just the bare inner CompiledModel.
    auto outer_inputs = inputs();
    auto outer_outputs = outputs();
    GQACompiledModel::write_port_list(stream, outer_inputs);
    GQACompiledModel::write_port_list(stream, outer_outputs);
    write(stream, m_dynamic_kv_cache_axes);
    write(stream, m_effective_max_context_len);
    write(stream, m_effective_sliding_window_capacity);
    // Written here (from the real model, available at this point) rather than
    // re-derived by import_model() from its dummy carcass outer_model, which has no
    // actual GroupQueryAttention ops for the graph-walking find_*() helpers to find.
    write(stream, m_total_seq_len_param_name);
    write(stream, m_present_to_past_param_names);
    GQA_TRACE("GQACompiledModel::export_model() -> wrote "
              << outer_inputs.size() << " outer inputs, " << outer_outputs.size() << " outer outputs, "
              << m_dynamic_kv_cache_axes.size()
              << " dynamic-axis entries, effective_max_context_len=" << m_effective_max_context_len
              << ", effective_sliding_window_capacity=" << m_effective_sliding_window_capacity
              << ", total_seq_len_param_name=" << m_total_seq_len_param_name.value_or("<none>")
              << ", present_to_past_param_names has " << m_present_to_past_param_names.size() << " entries"
              << "; delegating to inner m_inner_compiled_model->export_model()");
    m_inner_compiled_model->export_model(stream);
    LOG_INFO("Done");
    GQA_TRACE("GQACompiledModel::export_model() done");
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::GQACompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_INFO("Deserializing GQACompiledModel...");
    LOG_BLOCK();
    GQA_TRACE("GQACompiledModel::import_model() begin");

    using namespace ov::npuw::s11n;

    read_and_check_header(stream, NPUW_GQA_COMPILED_MODEL_INDICATOR, "GQACompiledModel");

    // Rebuild the outer-facing model (dynamic axes restored) exactly as before export.
    ov::ParameterVector outer_parameters;
    ov::NodeVector outer_results;
    std::unordered_map<std::string, size_t> dynamic_kv_cache_axes;
    uint32_t effective_max_context_len = 0;
    uint32_t effective_sliding_window_capacity = 0;
    std::optional<std::string> total_seq_len_param_name;
    std::unordered_map<std::string, std::string> present_to_past_param_names;
    outer_parameters = GQACompiledModel::read_input_port_list(stream);
    outer_results = GQACompiledModel::read_output_port_list(stream);
    read(stream, dynamic_kv_cache_axes);
    read(stream, effective_max_context_len);
    read(stream, effective_sliding_window_capacity);
    read(stream, total_seq_len_param_name);
    read(stream, present_to_past_param_names);

    GQA_TRACE("    read " << outer_parameters.size() << " outer parameters, " << outer_results.size()
                          << " outer results, " << dynamic_kv_cache_axes.size() << " dynamic-axis entries:");
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    for (const auto& [name, axis] : dynamic_kv_cache_axes) {
        GQA_TRACE("        '" << name << "' -> axis " << axis);
    }
#endif

    auto outer_model =
        std::make_shared<ov::Model>(ov::as_output_vector(outer_results), outer_parameters, "gqa_outer_model");

    // The rest of the stream is the self-contained inner CompiledModel ORC blob.
    auto inner_compiled_model = ov::npuw::CompiledModel::import_model(stream, plugin, properties);
    NPUW_ASSERT(inner_compiled_model != nullptr);
    GQA_TRACE("    inner CompiledModel imported @ "
              << inner_compiled_model.get()
              << "; wrapping it back into a GQACompiledModel so dynamic KV-cache/attention-bias reshape "
                 "survives the cache hit");

    return std::shared_ptr<GQACompiledModel>(new GQACompiledModel(outer_model,
                                                                  plugin,
                                                                  std::move(inner_compiled_model),
                                                                  std::move(dynamic_kv_cache_axes),
                                                                  effective_max_context_len,
                                                                  effective_sliding_window_capacity,
                                                                  std::move(total_seq_len_param_name),
                                                                  std::move(present_to_past_param_names)));
}

std::shared_ptr<const ov::Model> ov::npuw::GQACompiledModel::get_runtime_model() const {
    return m_outer_model;
}

void ov::npuw::GQACompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner_compiled_model->set_property(properties);
}

ov::Any ov::npuw::GQACompiledModel::get_property(const std::string& name) const {
    // NPUW_LLM_MAX_CONTEXT_LEN/NPUW_LLM_SWA_LEN are stripped from the inner model's
    // properties (ROOT-only options); answer them directly from the effective values
    // actually used.
    if (name == std::string(::intel_npu::NPUW_LLM_MAX_CONTEXT_LEN::key())) {
        return m_effective_max_context_len;
    }
    if (name == std::string(::intel_npu::NPUW_LLM_SWA_LEN::key())) {
        return m_effective_sliding_window_capacity;
    }
    return m_inner_compiled_model->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::GQACompiledModel::create_sync_infer_request() const {
    GQA_TRACE("GQACompiledModel::create_sync_infer_request() called");
    auto self = std::static_pointer_cast<const GQACompiledModel>(shared_from_this());
    auto request = std::make_shared<ov::npuw::GQAInferRequest>(std::move(self));
    GQA_TRACE("GQACompiledModel::create_sync_infer_request() -> GQAInferRequest created @ " << request.get());
    return request;
}

ov::npuw::GQAInferRequest::GQAInferRequest(std::shared_ptr<const GQACompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(std::move(compiled_model)) {
    GQA_TRACE("GQAInferRequest::GQAInferRequest() ctor, dynamic-axis map has "
              << m_compiled_model->m_dynamic_kv_cache_axes.size() << " entries");
}

void ov::npuw::GQAInferRequest::ensure_inner_request_locked() const {
    if (m_inner_request == nullptr) {
        m_inner_request = m_compiled_model->m_inner_compiled_model->create_infer_request();
        OPENVINO_ASSERT(m_inner_request != nullptr, "GQA infer request requires a valid inner request");
        const auto& inner_model = m_compiled_model->m_inner_compiled_model;
        GQA_TRACE("ensure_inner_request_locked(): created inner infer request @ "
                  << m_inner_request.get() << ", inner model has " << inner_model->inputs().size() << " inputs, "
                  << inner_model->outputs().size() << " outputs");

        // Zero-initialize every dynamic-axis KV-cache/bias input's full capacity, once.
        [[maybe_unused]] size_t zeroed_count = 0;  // only read by the developer-build GQA_TRACE below
        for (const auto& kv : m_compiled_model->m_dynamic_kv_cache_axes) {
            // Pre-C++20: structured bindings can't be captured by lambdas, so use a named local.
            const auto& name = kv.first;
            auto input_it = std::find_if(inner_model->inputs().begin(),
                                         inner_model->inputs().end(),
                                         [&name](const ov::Output<const ov::Node>& input) {
                                             return input.get_node()->get_friendly_name() == name;
                                         });
            if (input_it == inner_model->inputs().end()) {
                continue;
            }
            auto tensor = m_inner_request->get_tensor(*input_it);
            if (!tensor || tensor->get_byte_size() == 0) {
                continue;
            }
            std::memset(tensor->data(), 0, tensor->get_byte_size());
            ++zeroed_count;
        }
        GQA_TRACE("ensure_inner_request_locked(): zero-initialized " << zeroed_count << " dynamic-axis input(s)");
    }
}

// FIXME: Online search every time; a static mapping could be built once.
const ov::Output<const ov::Node>& ov::npuw::GQAInferRequest::map_port_locked(
    const ov::Output<const ov::Node>& port) const {
    ensure_inner_request_locked();

    const auto& outer_inputs = m_compiled_model->inputs();
    const auto& inner_inputs = m_inner_request->get_compiled_model()->inputs();
    for (size_t i = 0; i < outer_inputs.size(); ++i) {
        if (outer_inputs[i] == port) {
            OPENVINO_ASSERT(i < inner_inputs.size(), "Input port index is out of range in inner infer request");
            return inner_inputs[i];
        }
    }

    const auto& outer_outputs = m_compiled_model->outputs();
    const auto& inner_outputs = m_inner_request->get_compiled_model()->outputs();
    for (size_t i = 0; i < outer_outputs.size(); ++i) {
        if (outer_outputs[i] == port) {
            OPENVINO_ASSERT(i < inner_outputs.size(), "Output port index is out of range in inner infer request");
            return inner_outputs[i];
        }
    }

    OPENVINO_THROW("Unknown GQA infer request port: ", port.get_any_name());
}

// Diagnostic-only: dumps every known sequence-length-style input right before infer().
// Names are probed since exporters name/omit them differently.
void ov::npuw::GQAInferRequest::trace_sequence_length_inputs_locked() const {
    if (!gqa_trace_enabled()) {
        return;
    }
    static constexpr std::array<const char*, 3> kCandidateNames{"seqlens_k", "past_seq_len", "total_seq_len"};
    const auto& outer_inputs = m_compiled_model->inputs();
    for (const auto* candidate : kCandidateNames) {
        auto it = std::find_if(outer_inputs.begin(), outer_inputs.end(), [&](const auto& input) {
            return input.get_node()->get_friendly_name() == candidate;
        });
        if (it == outer_inputs.end()) {
            continue;
        }
        const auto& tensor = m_inner_request->get_tensor(map_port_locked(*it));
        [[maybe_unused]] const auto value = ov::npuw::read_scalar_int_tensor(tensor);
        GQA_TRACE("pre-infer sequence-length input '" << candidate << "' shape=" << tensor->get_shape() << " value="
                                                      << (value ? std::to_string(*value) : std::string("<n/a>")));
    }
}

// Diagnostic-only: dumps a min/max/non-zero-count summary for every dynamic-axis input
// that isn't a KV-cache tensor (i.e. the attention bias/mask).
void ov::npuw::GQAInferRequest::trace_attention_mask_stats_locked() const {
    if (!gqa_trace_enabled()) {
        return;
    }
    const auto& outer_inputs = m_compiled_model->inputs();
    for (const auto& kv : m_compiled_model->m_dynamic_kv_cache_axes) {
        // Pre-C++20: structured bindings can't be captured by lambdas, so use named locals.
        const auto& name = kv.first;
        [[maybe_unused]] const auto axis = kv.second;  // only referenced by GQA_TRACE below
        if (ov::npuw::util::contains_ignore_case(name, "past_key") ||
            ov::npuw::util::contains_ignore_case(name, "past_value")) {
            continue;  // KV-cache tensor, not the mask/bias -- already traced/handled elsewhere
        }
        auto it = std::find_if(outer_inputs.begin(), outer_inputs.end(), [&name](const auto& input) {
            return input.get_node()->get_friendly_name() == name;
        });
        if (it == outer_inputs.end()) {
            continue;
        }
        const auto& tensor = m_inner_request->get_tensor(map_port_locked(*it));
        GQA_TRACE("pre-infer attention-bias/mask input '" << name << "' axis=" << axis
                                                          << " shape=" << tensor->get_shape() << " "
                                                          << ov::npuw::tensor_stats_string(tensor));
    }
}

void ov::npuw::GQAInferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    sync_dynamic_kv_cache_tensors_locked();
    trace_sequence_length_inputs_locked();
    trace_attention_mask_stats_locked();
    GQA_TRACE("GQAInferRequest::infer() -> delegating to inner request @ " << m_inner_request.get());
    // TODO: enclose with try/catch only when tracing is enabled.
    try {
        m_inner_request->infer();
    } catch (const std::exception& ex) {
        // Surface the real exception message before it propagates and gets wrapped away.
        (void)ex;  // only referenced by GQA_TRACE, a no-op outside NPU_PLUGIN_DEVELOPER_BUILD
        GQA_TRACE("GQAInferRequest::infer() -> inner request infer() THREW: " << ex.what());
        throw;
    } catch (...) {
        GQA_TRACE("GQAInferRequest::infer() -> inner request infer() THREW an unknown exception");
        throw;
    }
    GQA_TRACE("GQAInferRequest::infer() -> inner request infer() returned");
    refresh_present_tensors_locked();
}

// Reads the live value of the GQA op's TOTAL_SEQUENCE_LENGTH Parameter from the inner
// request's tensor. Returns nullopt if unresolved or not a readable scalar.
std::optional<size_t> ov::npuw::GQAInferRequest::read_total_sequence_length_locked() const {
    const auto& param_name = m_compiled_model->m_total_seq_len_param_name;
    if (!param_name) {
        return std::nullopt;
    }
    const auto& outer_inputs = m_compiled_model->inputs();
    auto port_it = std::find_if(outer_inputs.begin(), outer_inputs.end(), [&](const auto& input) {
        return input.get_node()->get_friendly_name() == *param_name;
    });
    if (port_it == outer_inputs.end()) {
        return std::nullopt;
    }
    const auto& tensor = m_inner_request->get_tensor(map_port_locked(*port_it));
    auto value = ov::npuw::read_scalar_int_tensor(tensor);
    if (!value || *value < 0) {
        return std::nullopt;
    }
    return static_cast<size_t>(*value);
}

void ov::npuw::GQAInferRequest::copy_kv_cache_prefix(const ov::SoPtr<ov::ITensor>& src,
                                                     const ov::SoPtr<ov::ITensor>& dst,
                                                     size_t axis) {
    OPENVINO_ASSERT(src->get_element_type() == dst->get_element_type());
    const auto& src_shape = src->get_shape();
    const auto& dst_shape = dst->get_shape();
    OPENVINO_ASSERT(src_shape.size() == 4u && dst_shape.size() == 4u, "Expected rank-4 KV-cache tensors");
    OPENVINO_ASSERT(axis == 2 || axis == 3, "Unsupported dynamic KV-cache axis ", axis);
    if (axis == 2) {
        // Axis 2 (not transposed): reuse the per-plane KV-cache copy helper.
        ov::npuw::util::copy_by_planes(src, dst);
        return;
    }
    // Axis 3 (transposed V-cache): copy each row's valid-length prefix, left-aligned.
    OPENVINO_ASSERT(src_shape[0] == 1u, "Expected batch size 1");
    OPENVINO_ASSERT(src_shape[0] == dst_shape[0] && src_shape[1] == dst_shape[1] && src_shape[2] == dst_shape[2]);
    const auto* src_p = reinterpret_cast<const uint8_t*>(src->data());
    auto* dst_p = reinterpret_cast<uint8_t*>(dst->data());
    const auto rows = src_shape[1] * src_shape[2];
    const auto src_row_stride = src->get_strides()[2];
    const auto dst_row_stride = dst->get_strides()[2];
    // Use the tensors' reported shape, not get_strides(): a ROI view's stride reflects
    // the parent's full row length, not the (possibly shorter) valid length.
    const auto row_bytes = std::min(src_shape[3], dst_shape[3]) * src->get_element_type().size();
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(src_p, row_bytes, dst_p);
        src_p += src_row_stride;
        dst_p += dst_row_stride;
    }
}

// Refreshes outer present_key/present_value tensors after infer(): copies the inner
// buffer's valid prefix into a dense tensor, or aliases it directly when the whole
// buffer is already valid and no copy is needed (see m_dynamic_kv_cache_output_aliased).
//
// Each output name is in one of 3 states across infer() calls: (1) never allocated yet,
// (2) aliased -- m_dynamic_kv_cache_output_tensors[name] IS the inner request's own live
// buffer (valid_len == capacity, nothing to exclude), or (3) private -- a separate,
// densely-shaped buffer holding just the valid prefix (valid_len < capacity). The state
// can flip either way between calls as valid_len/capacity change. Below,
// m_dynamic_kv_cache_output_aliased.erase(name) reports (via its return value) whether
// state (2) applied a moment ago: if so, the cached tensor is literally the inner
// buffer and must be discarded and freshly allocated rather than reused, since writing
// through it would corrupt the inner request's working buffer; otherwise it is already
// a private buffer and can simply be resized in place, keeping its identity stable for
// callers that cache the tensor pointer/object.
void ov::npuw::GQAInferRequest::refresh_present_tensors_locked() const {
    const auto& output_axes = m_compiled_model->m_dynamic_kv_cache_output_axes;
    if (output_axes.empty()) {
        return;
    }

    const auto& outer_outputs = m_compiled_model->outputs();
    for (const auto& kv : output_axes) {
        // Pre-C++20: structured bindings can't be captured by lambdas, so use named locals.
        const auto& name = kv.first;
        const auto axis = kv.second;
        auto port_it = std::find_if(outer_outputs.begin(), outer_outputs.end(), [&](const auto& output) {
            return output.get_node()->get_friendly_name() == name;
        });
        if (port_it == outer_outputs.end()) {
            continue;
        }
        if (m_dynamic_kv_cache_output_externally_set.count(name) != 0) {
            // set_tensor() forwarded this exact buffer to the inner request (shape is
            // asserted to match its compiled capacity); the inner request already wrote
            // directly into it during infer(), nothing to copy.
            continue;
        }
        const auto inner_tensor = m_inner_request->get_tensor(map_port_locked(*port_it));
        const auto capacity = inner_tensor->get_shape().at(axis);

        // Present length = max(past_len, total_sequence_length), not just past_len (which
        // misses rows just appended). Present->past mapping comes from the GQA op's own
        // graph connections, not string rewriting (breaks for dot-style naming).
        // total_sequence_length is read independently of past_len -- the caller may
        // legitimately never have touched the past_key/value input at all (e.g. it
        // queries every output tensor up front, before any set_tensor()) and
        // total_sequence_length alone is still enough to answer accurately instead of
        // silently falling back to the full, un-copied capacity.
        std::optional<size_t> past_len;
        const auto& present_to_past = m_compiled_model->m_present_to_past_param_names;
        if (auto past_name_it = present_to_past.find(name); past_name_it != present_to_past.end()) {
            if (auto past_it = m_dynamic_kv_cache_tensors.find(past_name_it->second);
                past_it != m_dynamic_kv_cache_tensors.end()) {
                past_len = past_it->second->get_shape().at(axis);
            }
        }
        const auto total_len = read_total_sequence_length_locked();
        size_t valid_len = capacity;
        if (past_len || total_len) {
            valid_len = std::min<size_t>(capacity, std::max<size_t>(past_len.value_or(0u), total_len.value_or(0u)));
        }

        if (valid_len == capacity) {
            // Whole buffer valid, no copy needed: alias the outer tensor onto the inner
            // buffer directly. NEVER set_shape() this entry while aliased -- it's the
            // inner request's own live working buffer.
            m_dynamic_kv_cache_output_tensors[name] = inner_tensor;
            m_dynamic_kv_cache_output_aliased.insert(name);
            continue;
        }

        auto out_shape = inner_tensor->get_shape();
        out_shape.at(axis) = valid_len;

        auto& outer_tensor = m_dynamic_kv_cache_output_tensors[name];
        const bool was_aliased = m_dynamic_kv_cache_output_aliased.erase(name) != 0;
        if (!outer_tensor || was_aliased) {
            // Never allocated, or aliased onto the inner tensor: allocate a fresh,
            // privately-owned buffer instead of resizing in place.
            //
            // Investigated using the inner model's own device context here (the same
            // remote/host-tensor mechanism as the weights bank, via
            // ov::npuw::util::allocMem()) instead of a plain host allocation. Measured
            // as a ~2x decode-throughput regression end-to-end: unlike weights (written
            // once, read many times), this buffer is rewritten by copy_kv_cache_prefix()
            // on every refresh, and repeated host-side writes into NPU-host-accessible
            // memory were far slower than into plain process memory. Kept as a plain
            // CPU allocation.
            outer_tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(inner_tensor->get_element_type(), out_shape));
        } else if (outer_tensor->get_shape() != out_shape) {
            // Resize in place to keep this tensor's identity stable across calls (cached
            // IO-binding callers may hold onto the returned object/data pointer).
            outer_tensor->set_shape(out_shape);
        }

        // A strided ROI view over the inner buffer's valid prefix, only used internally
        // for copy_kv_cache_prefix -- never handed to the caller.
        ov::Coordinate begin(inner_tensor->get_shape().size(), 0);
        ov::Coordinate end(inner_tensor->get_shape());
        end.at(axis) = valid_len;
        auto inner_view = ov::SoPtr<ov::ITensor>(ov::make_tensor(inner_tensor._ptr, begin, end));
        ov::npuw::GQAInferRequest::copy_kv_cache_prefix(inner_view, outer_tensor, axis);
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::GQAInferRequest::get_present_tensor_locked(const std::string& name) const {
    auto it = m_dynamic_kv_cache_output_tensors.find(name);
    if (it != m_dynamic_kv_cache_output_tensors.end()) {
        return it->second;
    }
    // infer() hasn't run yet; ORT/OVEP may query output tensors before the first infer().
    // Fall back to the inner request's raw tensor instead of hard-failing.
    GQA_TRACE("get_present_tensor_locked('"
              << name << "'): no refreshed tensor yet (infer() not called?), falling back to inner request tensor");
    const auto& outer_outputs = m_compiled_model->outputs();
    auto port_it = std::find_if(outer_outputs.begin(), outer_outputs.end(), [&](const auto& output) {
        return output.get_node()->get_friendly_name() == name;
    });
    OPENVINO_ASSERT(port_it != outer_outputs.end(), "GQA present output '", name, "' port not found");
    return m_inner_request->get_tensor(map_port_locked(*port_it));
}

ov::SoPtr<ov::ITensor> ov::npuw::GQAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    const auto& name = port.get_node()->get_friendly_name();
    GQA_TRACE("GQAInferRequest::get_tensor(port='" << name << "')");
    if (m_compiled_model->m_dynamic_kv_cache_axes.count(name) != 0) {
        // The inner tensor is the static-capacity buffer, not what the caller set; hand
        // back the exact user-owned tensor instead.
        auto it = m_dynamic_kv_cache_tensors.find(name);
        if (it == m_dynamic_kv_cache_tensors.end()) {
            // get_tensor() before set_tensor() (IO-binding pattern): allocate a private
            // tensor. Dynamic dims are zeroed (static dims keep real extent), matching
            // intel_cpu's convention for a never-set dynamic tensor -- a full-capacity
            // shape would wrongly look "already populated". Caller must set_shape()
            // before writing.
            const auto& inner_tensor = m_inner_request->get_tensor(map_port_locked(port));
            const auto& port_shape = port.get_partial_shape();
            ov::Shape private_shape;
            private_shape.reserve(port_shape.size());
            for (const auto& dim : port_shape) {
                private_shape.push_back(dim.is_static() ? dim.get_length() : 0u);
            }
            // Same reasoning as refresh_present_tensors_locked()'s fresh present buffer:
            // investigated device-context allocation here too, but kept as a plain CPU
            // allocation since the host-write path is what matters, not device-read
            // locality (see that function's comment for the measured regression).
            auto private_tensor =
                ov::SoPtr<ov::ITensor>(ov::make_tensor(inner_tensor->get_element_type(), private_shape));
            it = m_dynamic_kv_cache_tensors.emplace(name, private_tensor).first;
            GQA_TRACE(
                "    -> no set_tensor() yet, allocated private fallback tensor, shape=" << private_tensor->get_shape());
        }
        GQA_TRACE("    -> returning previously set_tensor()'d user tensor, shape=" << it->second->get_shape());
        return it->second;
    }

    if (auto out_it = m_compiled_model->m_dynamic_kv_cache_output_axes.find(name);
        out_it != m_compiled_model->m_dynamic_kv_cache_output_axes.end()) {
        auto tensor = get_present_tensor_locked(name);
        GQA_TRACE("    -> returning present output tensor, shape=" << tensor->get_shape());
        return tensor;
    }

    auto tensor = m_inner_request->get_tensor(map_port_locked(port));
    GQA_TRACE("    -> returning inner request tensor, shape=" << tensor->get_shape());
    return tensor;
}

void ov::npuw::GQAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                           const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    const auto& name = port.get_node()->get_friendly_name();
    GQA_TRACE("GQAInferRequest::set_tensor(port='" << name << "', shape=" << tensor->get_shape() << ")");

    // Mark present_* outputs externally owned so refresh_present_tensors_locked() won't
    // later republish a diverging, separately-copied tensor under this name.
    if (m_compiled_model->m_dynamic_kv_cache_output_axes.count(name) != 0) {
        GQA_TRACE("    -> dynamic-axis present output, forwarding to inner request and marking externally owned");
        // The inner request is bound to a STATIC compiled model: its own set_tensor()
        // (IBaseInferRequest::set_tensor()) stores whatever it's given with zero shape
        // validation (check_tensors() is a no-op too), yet the model can only ever
        // correctly produce/consume its one compiled shape. Fail early instead of
        // letting a wrong-sized buffer reach the inner request unchecked. (Caller-driven
        // dynamic present-output sizing below the compiled capacity is a separate,
        // more involved problem -- tracked separately, not handled here.)
        const auto& inner_tensor = m_inner_request->get_tensor(map_port_locked(port));
        OPENVINO_ASSERT(tensor->get_shape() == inner_tensor->get_shape(),
                        "GQA present output '",
                        name,
                        "' shape ",
                        tensor->get_shape(),
                        " does not match the compiled static shape ",
                        inner_tensor->get_shape());
        m_inner_request->set_tensor(map_port_locked(port), tensor);
        m_dynamic_kv_cache_output_tensors[name] = tensor;
        m_dynamic_kv_cache_output_externally_set.insert(name);
        return;
    }

    const auto& dynamic_axes = m_compiled_model->m_dynamic_kv_cache_axes;
    auto it = dynamic_axes.find(name);
    if (it == dynamic_axes.end()) {
        GQA_TRACE("    -> not a dynamic-axis input, forwarding to inner request as-is");
        m_inner_request->set_tensor(map_port_locked(port), tensor);
        return;
    }

    // If the caller's tensor is EXACTLY the compiled static capacity, there's no prefix
    // to carve out: alias the inner request directly onto it and skip the copy.
    // Otherwise keep the inner tensor and defer the prefix copy to infer() time.
    const auto axis = it->second;
    const auto& inner_tensor = m_inner_request->get_tensor(map_port_locked(port));
    const auto requested_len = tensor->get_shape().at(axis);
    const auto capacity = inner_tensor->get_shape().at(axis);
    GQA_TRACE("    -> dynamic-axis input, axis=" << axis << " requested_len=" << requested_len << " capacity="
                                                 << capacity << " inner_tensor_shape=" << inner_tensor->get_shape());
    OPENVINO_ASSERT(requested_len <= capacity,
                    "GQA KV-cache '",
                    name,
                    "' length ",
                    requested_len,
                    " exceeds the static capacity (",
                    capacity,
                    ") it was compiled with");
    if (requested_len == capacity) {
        // Exact match: alias instead of copying, nothing to carve out.
        if (m_dynamic_kv_cache_input_aliased.count(name) == 0) {
            // First alias of this port: stash the inner request's own buffer so a later,
            // shorter set_tensor() can restore it.
            m_dynamic_kv_cache_private_buffers[name] = inner_tensor;
        }
        m_inner_request->set_tensor(map_port_locked(port), tensor);
        m_dynamic_kv_cache_tensors[name] = tensor;
        m_dynamic_kv_cache_input_aliased.insert(name);
        GQA_TRACE("    -> requested_len == capacity, aliased inner request tensor directly (no copy)");
        return;
    }
    if (m_dynamic_kv_cache_input_aliased.erase(name) != 0) {
        // Previously aliased onto a now-stale caller tensor; restore the inner request's
        // own private buffer before this shorter tensor is used.
        auto priv_it = m_dynamic_kv_cache_private_buffers.find(name);
        OPENVINO_ASSERT(priv_it != m_dynamic_kv_cache_private_buffers.end(),
                        "GQA KV-cache '",
                        name,
                        "' was aliased but has no stashed private buffer to restore");
        m_inner_request->set_tensor(map_port_locked(port), priv_it->second);
        GQA_TRACE("    -> requested_len < capacity, restored inner request's own private buffer");
    }
    // Not copied here: the caller may keep writing into `tensor` until infer(); only
    // sync_dynamic_kv_cache_tensors_locked() is guaranteed to see live content.
    m_dynamic_kv_cache_tensors[name] = tensor;
    GQA_TRACE("    -> stored user tensor reference for later sync at infer() time");
}

void ov::npuw::GQAInferRequest::sync_dynamic_kv_cache_tensors_locked() const {
    const auto& outer_inputs = m_compiled_model->inputs();
    for (const auto& kv : m_compiled_model->m_dynamic_kv_cache_axes) {
        // Pre-C++20: structured bindings can't be captured by lambdas, so use named locals.
        const auto& name = kv.first;
        const auto axis = kv.second;
        auto tensor_it = m_dynamic_kv_cache_tensors.find(name);
        if (tensor_it == m_dynamic_kv_cache_tensors.end()) {
            continue;  // set_tensor() not called yet for this port
        }
        auto port_it = std::find_if(outer_inputs.begin(), outer_inputs.end(), [&](const auto& input) {
            return input.get_node()->get_friendly_name() == name;
        });
        if (port_it == outer_inputs.end()) {
            continue;
        }
        const auto& inner_tensor = m_inner_request->get_tensor(map_port_locked(*port_it));
        if (&*tensor_it->second == &*inner_tensor) {
            // Already aliased by set_tensor(); same object, nothing to copy.
            GQA_TRACE("sync_dynamic_kv_cache_tensors_locked(): '" << name << "' already aliased, skipping copy");
            continue;
        }
        GQA_TRACE("sync_dynamic_kv_cache_tensors_locked(): '"
                  << name << "' axis=" << axis << " copying live prefix, shape=" << tensor_it->second->get_shape());
        // Re-check here: the tensor can still grow between set_tensor() (which checked
        // <= capacity at the time) and infer() -- e.g. a private fallback from
        // get_tensor() followed by set_shape(), or a caller tensor reshaped afterwards.
        // copy_by_planes() (axis 2) sizes each plane from the source, so an unchecked
        // over-capacity tensor here would write past the end of the static buffer.
        const auto src_len = tensor_it->second->get_shape().at(axis);
        const auto capacity = inner_tensor->get_shape().at(axis);
        OPENVINO_ASSERT(src_len <= capacity,
                        "GQA KV-cache '",
                        name,
                        "' length ",
                        src_len,
                        " exceeds the static capacity (",
                        capacity,
                        ") it was compiled with");
        ov::npuw::GQAInferRequest::copy_kv_cache_prefix(tensor_it->second, inner_tensor, axis);
    }
}

void ov::npuw::GQAInferRequest::check_tensors() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    // Trigger lazy inner request initialization; the JustInferRequest constructor
    // allocates all sub-tensors during construction, so nothing more is needed here.
    ensure_inner_request_locked();
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::GQAInferRequest::query_state() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_inner_request->query_state();
}

std::vector<ov::ProfilingInfo> ov::npuw::GQAInferRequest::get_profiling_info() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_inner_request->get_profiling_info();
}
