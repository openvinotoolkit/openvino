// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa_compiled_model.hpp"

#include <algorithm>
#include <array>
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

// Reads a single NPUW option out of a raw (not yet Config-wrapped) properties AnyMap,
// falling back to the option's own default when absent. Mirrors the cfg_get<>() helper
// in compiled_model.cpp/flux2_compiled_model.cpp.
template <typename T>
auto cfg_get(const ov::AnyMap& properties) -> typename T::ValueType {
    const auto& opt_name = std::string(T::key());
    if (properties.count(opt_name)) {
        return properties.at(opt_name).as<typename T::ValueType>();
    }
    return T::defaultValue();
}

// True if GQA-specific diagnostic tracing (the "[GQA-TRACE] ..." lines throughout this
// file) is enabled via the OPENVINO_NPUW_GQA_LOG=1 environment variable. Checked once and
// cached; intentionally a separate on/off knob from the general NPUW_LOG_LEVEL, since
// this tracing is extremely verbose (dumps tensor stats on every infer()) and exists
// purely to support this GQA accuracy investigation.
bool gqa_trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("OPENVINO_NPUW_GQA_LOG");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
}

// Scans `model`'s past_key/past_value Parameters for a dynamic dimension and returns the
// axis to pin to the configured static capacity (NPUW_LLM_MAX_CONTEXT_LEN), keyed by
// Parameter friendly name. A KV-cache Parameter with more than one dynamic dimension is
// ambiguous and not something we can safely resolve.
std::unordered_map<std::string, size_t> find_dynamic_kv_cache_axes(const std::shared_ptr<const ov::Model>& model) {
    std::unordered_map<std::string, size_t> result;

    const auto resolve_dynamic_axis =
        [](const std::shared_ptr<ov::op::v0::Parameter>& parameter) -> std::optional<size_t> {
        const auto& partial_shape = parameter->get_partial_shape();
        if (partial_shape.rank().is_dynamic()) {
            return std::nullopt;
        }
        std::optional<size_t> dynamic_axis;
        for (size_t i = 0; i < partial_shape.size(); ++i) {
            if (partial_shape[i].is_dynamic()) {
                OPENVINO_ASSERT(!dynamic_axis.has_value(),
                                "GQA parameter '",
                                parameter->get_friendly_name(),
                                "' has more than one dynamic dimension; can't resolve its max_seq_len axis");
                dynamic_axis = i;
            }
        }
        return dynamic_axis;
    };

    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (!ov::npuw::util::contains_ignore_case(name, "past_key") &&
            !ov::npuw::util::contains_ignore_case(name, "past_value")) {
            continue;
        }
        if (auto axis = resolve_dynamic_axis(parameter)) {
            result.emplace(name, *axis);
        }
    }

    // The attention bias/mask shares the KV-cache's max_seq_len dimension but isn't named
    // consistently across exporters, so it's located via the GQA op's ATTENTION_BIAS input
    // instead of by Parameter name.
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa || gqa->get_input_size() <= static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)) {
            continue;
        }
        auto bias_parameter = ov::as_type_ptr<ov::op::v0::Parameter>(
            gqa->input_value(static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)).get_node_shared_ptr());
        if (!bias_parameter) {
            continue;  // not fed directly by a Parameter; nothing we can reshape here
        }
        if (auto axis = resolve_dynamic_axis(bias_parameter)) {
            result.emplace(bias_parameter->get_friendly_name(), *axis);
        }
    }

    return result;
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

std::pair<ov::AnyMap, GQAModelStage> with_gqa_defaults(const std::shared_ptr<ov::Model>& model,
                                                       const ov::AnyMap& properties) {
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

    // with_gqa_defaults() only runs once identify_case() already confirmed V0 or V1,
    // so online partitioning is disabled for both.
    ov::AnyMap config = {
        {"NPUW_ONLINE_PIPELINE", "NONE"},
        {std::string(::intel_npu::NPUW_DEVICES::key()), "NPU"},
        {ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SPEED},
        {std::string(::intel_npu::NPUW_UNQDQ::key()), "YES"},
        // {"NPU_COMPILER_TYPE", "DRIVER"},
        // {"LOG_LEVEL", "LOG_INFO"}
    };

    const auto stage = detect_gqa_model_stage();
    if (stage == GQAModelStage::PREFILL) {
        merge_config_with(config,
                          {{std::string(::intel_npu::NPUW_FOLD::key()), "YES"},
                           {"NPUW_FOLD_ONLY", "attn"},
                           {"NPUW_ONLINE_ISOLATE", "ATTN"},
                           {"NPUW_ONLINE_KEEP_BLOCKS_TAGGED", "attn"},
                           {"NPUW_ATTN", "STATIC"}});
        LOG_INFO("Detected prefill-style GQA model; applying FOLD with ATTN isolation");
    } else if (stage == GQAModelStage::GENERATE) {
        merge_config_with(config,
                          {{std::string(::intel_npu::NPUW_FOLD::key()), "YES"},
                           {"NPUW_FOLD_ONLY", "attn"},
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
    return {config, stage};
}

}  // namespace

// Prints a GQA diagnostic trace line, prefixed with "[GQA-TRACE] ", only if
// OPENVINO_NPUW_GQA_LOG=1 is set in the environment (see gqa_trace_enabled() above); a
// no-op otherwise. Mirrors the LOG_INFO/LOG_DEBUG/LOG_VERB style in logging.hpp: `msg` is
// a chain of `<<`-streamed expressions, e.g. GQA_TRACE("value=" << value << " shape=" << shape).
#define GQA_TRACE(msg)                                       \
    do {                                                     \
        if (gqa_trace_enabled()) {                           \
            std::cout << "[GQA-TRACE] " << msg << std::endl; \
        }                                                    \
    } while (0)

ov::npuw::GQACompiledModel::PreparedState ov::npuw::GQACompiledModel::prepare(const std::shared_ptr<ov::Model>& model,
                                                                              const ov::AnyMap& properties) {
    auto [prepared_properties, stage] = with_gqa_defaults(model, properties);

    static std::size_t mcount = 0u;

    model->set_friendly_name(model->get_friendly_name() + "_gqa_" +
                             (stage == GQAModelStage::PREFILL    ? "prefill"
                              : stage == GQAModelStage::GENERATE ? "generate"
                                                                 : "unknown") +
                             std::to_string(mcount++));
    // ov::save_model(model, model->get_friendly_name() + ".xml");

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
    if (has_dynamic_max_seq_len(model)) {
        // Static capacity (in tokens) a dynamic KV-cache is reshaped to for NPU
        // compilation. Configurable via NPUW_LLM_MAX_CONTEXT_LEN; falls back to that
        // option's own default (8192) if it isn't present in `properties`.
        const size_t max_seq_len = cfg_get<::intel_npu::NPUW_LLM_MAX_CONTEXT_LEN>(properties);
        dynamic_kv_cache_axes = find_dynamic_kv_cache_axes(model);
        OPENVINO_ASSERT(!dynamic_kv_cache_axes.empty(),
                        "GQA model has a dynamic max_seq_len but no resolvable KV-cache Parameter was found");
        compiled_model = model->clone();
        std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
        for (const auto& [name, axis] : dynamic_kv_cache_axes) {
            const auto& params = compiled_model->get_parameters();
            auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
                return parameter->get_friendly_name() == name;
            });
            OPENVINO_ASSERT(it != params.end(), "KV-cache parameter '", name, "' not found in the cloned model");
            auto new_shape = (*it)->get_partial_shape();
            new_shape[axis] = ov::Dimension(static_cast<int64_t>(max_seq_len));
            new_shapes[(*it)->output(0)] = new_shape;
        }
        compiled_model->reshape(new_shapes);
        for (const auto& [name, axis] : dynamic_kv_cache_axes) {
            const auto& params = compiled_model->get_parameters();
            auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
                return parameter->get_friendly_name() == name;
            });
            OPENVINO_ASSERT(it != params.end() && (*it)->get_partial_shape().is_static(),
                            "Reshaping the GQA KV-cache parameter '",
                            name,
                            "' to a static capacity of ",
                            max_seq_len,
                            " did not make it fully static");
        }
        LOG_INFO("Reshaped dynamic GQA KV-cache to a static capacity of " << max_seq_len << " tokens");
    }

    return {model, compiled_model, std::move(prepared_properties), std::move(dynamic_kv_cache_axes)};
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::GQACompiledModel::make_compiled_model(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    return std::make_shared<ov::npuw::CompiledModel>(model, plugin, properties);
}

namespace {

bool has_valid_gqa_op(const std::shared_ptr<const ov::Model>& model) {
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionInputs;
    constexpr size_t mandatory_inputs = static_cast<size_t>(GroupQueryAttentionInputs::TOTAL_SEQUENCE_LENGTH) + 1;

    for (const auto& node : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(node);
        if (!gqa) {
            continue;
        }
        const auto num_heads = gqa->get_num_heads();
        const auto kv_num_heads = gqa->get_kv_num_heads();
        if (num_heads <= 0 || kv_num_heads <= 0 || num_heads % kv_num_heads != 0 ||
            gqa->get_input_size() < mandatory_inputs) {
            continue;  // malformed instance, doesn't count as evidence
        }
        if (gqa->get_do_rotary() &&
            gqa->get_input_size() <= static_cast<size_t>(GroupQueryAttentionInputs::SIN_CACHE)) {
            continue;  // rotary requires the cos/sin cache inputs
        }
        return true;
    }
    return false;
}

}  // namespace

bool ov::npuw::GQACompiledModel::supports(const std::shared_ptr<const ov::Model>& model) {
    // identify_case() classifies the model's GQA family (V0/V1) purely from its structure,
    // independent of whether any dimension is dynamic. supports() is the auto-dispatch gate:
    // this wrapper only exists to bridge a *dynamic* max_seq_len KV-cache/attention-bias to
    // the NPU's static-shape requirement, so a model that is already fully static doesn't
    // need it, even if it otherwise matches a known GQA family.
    return identify_case(model) != Case::Unknown && has_dynamic_max_seq_len(model);
}

ov::npuw::GQACompiledModel::Case ov::npuw::GQACompiledModel::identify_case(
    const std::shared_ptr<const ov::Model>& model) {
    if (!has_valid_gqa_op(model)) {
        return Case::Unknown;
    }

    // V1 conveys RoPE position via an explicit `position_ids` input; V0 conveys it via
    // `past_seq_len`/`total_seq_len` with no `position_ids` input.
    bool has_activation_input = false;
    bool has_position_ids = false;
    bool has_seq_len_signal = false;
    bool has_past_kv_cache = false;
    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
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
        if (util::contains_ignore_case(name, "present_key") || util::contains_ignore_case(name, "present_value") ||
            util::contains_ignore_case(name, "present.")) {
            has_present_kv_cache = true;
            break;
        }
    }
    if (!has_present_kv_cache) {
        return Case::Unknown;
    }

    return has_position_ids ? Case::V1 : Case::V0;
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
            if (shape.rank().is_dynamic()) {
                continue;
            }
            for (const auto& dim : shape) {
                if (dim.is_dynamic()) {
                    return true;
                }
            }
        }
        // The attention bias/mask (if present) carries the same max_seq_len dimension as
        // the KV-cache and must be reshaped alongside it.
        if (gqa->get_input_size() > static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)) {
            const auto& bias_shape =
                gqa->input_value(static_cast<size_t>(GroupQueryAttentionInputs::ATTENTION_BIAS)).get_partial_shape();
            if (bias_shape.rank().is_static()) {
                for (const auto& dim : bias_shape) {
                    if (dim.is_dynamic()) {
                        return true;
                    }
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
        // The ONNX frontend appends "/sink_port_0" to a Result's friendly name (see
        // translate_session.cpp), so pattern-match against the part before the first '/'.
        const auto slash_pos = full_name.find('/');
        const auto bare_name = slash_pos == std::string::npos ? full_name : full_name.substr(0, slash_pos);
        if (!ov::npuw::util::contains_ignore_case(bare_name, "present_key") &&
            !ov::npuw::util::contains_ignore_case(bare_name, "present_value")) {
            continue;
        }
        const auto& shape = output.get_partial_shape();
        if (shape.rank().is_dynamic()) {
            continue;
        }
        std::optional<size_t> dynamic_axis;
        bool ambiguous = false;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i].is_dynamic()) {
                if (dynamic_axis.has_value()) {
                    ambiguous = true;
                    break;
                }
                dynamic_axis = i;
            }
        }
        if (ambiguous || !dynamic_axis.has_value()) {
            continue;  // no resolvable dynamic axis; leave it out, get_tensor() will fall back
        }
        result.emplace(full_name, *dynamic_axis);
    }
    return result;
}

std::optional<std::string> ov::npuw::GQACompiledModel::present_to_past_name(const std::string& name) {
    const auto slash_pos = name.find('/');
    const auto bare_name = slash_pos == std::string::npos ? name : name.substr(0, slash_pos);
    static const std::string kPresentToken = "present";
    auto it = std::search(bare_name.begin(),
                          bare_name.end(),
                          kPresentToken.begin(),
                          kPresentToken.end(),
                          [](unsigned char a, unsigned char b) {
                              return std::tolower(a) == std::tolower(b);
                          });
    if (it == bare_name.end()) {
        return std::nullopt;
    }
    const auto pos = static_cast<size_t>(std::distance(bare_name.begin(), it));
    return bare_name.substr(0, pos) + "past" + bare_name.substr(pos + kPresentToken.size());
}

void ov::npuw::GQACompiledModel::copy_kv_cache_prefix(const ov::SoPtr<ov::ITensor>& src,
                                                      const ov::SoPtr<ov::ITensor>& dst,
                                                      size_t axis) {
    OPENVINO_ASSERT(src->get_element_type() == dst->get_element_type());
    const auto& src_shape = src->get_shape();
    const auto& dst_shape = dst->get_shape();
    OPENVINO_ASSERT(src_shape.size() == 4u && dst_shape.size() == 4u, "Expected rank-4 KV-cache tensors");
    OPENVINO_ASSERT(axis == 2 || axis == 3, "Unsupported dynamic KV-cache axis ", axis);
    if (axis == 2) {
        // S is not the last dim (V-cache not transposed): reuse the existing per-plane
        // (N=1, iterate H, copy S*E contiguous elements) KV-cache copy helper.
        ov::npuw::util::copy_by_planes(src, dst);
        return;
    }
    // S is the last dim (transposed V-cache): each (n, h, e) row is a contiguous S-length
    // chunk; copy every row's valid-length prefix into the wider-capacity row, left-aligned.
    // This is the axis==3 analog of copy_by_planes, walking the same N=1/H/E "rows".
    OPENVINO_ASSERT(src_shape[0] == 1u, "Expected batch size 1");
    OPENVINO_ASSERT(src_shape[0] == dst_shape[0] && src_shape[1] == dst_shape[1] && src_shape[2] == dst_shape[2]);
    const auto* src_p = reinterpret_cast<const uint8_t*>(src->data());
    auto* dst_p = reinterpret_cast<uint8_t*>(dst->data());
    const auto rows = src_shape[1] * src_shape[2];
    const auto src_row_stride = src->get_strides()[2];
    const auto dst_row_stride = dst->get_strides()[2];
    // The number of bytes to copy per row must come from the tensors' *reported shape*
    // at the S axis (src_shape[3]/dst_shape[3]), not from get_strides()[2]: for a plain
    // dense tensor the two happen to coincide (the S axis is contiguous), but for a ROI
    // view over a larger parent buffer (see GQAInferRequest::refresh_present_tensors_locked)
    // the stride still reflects the *parent's* full row length while the reported shape
    // correctly reflects the ROI's (smaller) valid length -- using the stride here would
    // read/write past the shorter tensor's actual row.
    const auto row_bytes = std::min(src_shape[3], dst_shape[3]) * src->get_element_type().size();
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(src_p, row_bytes, dst_p);
        src_p += src_row_stride;
        dst_p += dst_row_stride;
    }
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
      m_dynamic_kv_cache_axes(std::move(prepared.dynamic_kv_cache_axes)) {
    OPENVINO_ASSERT(m_inner_compiled_model != nullptr, "GQACompiledModel requires a valid inner compiled model");
    m_dynamic_kv_cache_output_axes = find_dynamic_kv_cache_output_axes(outputs());
}

// Used by import_model() only: wraps an already-deserialized inner compiled model
// (no factory/compilation involved) together with the outer-facing model rebuilt
// from the exported ports and the dynamic-axis map, so a cache hit produces a fully
// functional GQACompiledModel rather than a bare inner CompiledModel.
ov::npuw::GQACompiledModel::GQACompiledModel(const std::shared_ptr<ov::Model>& outer_model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             std::shared_ptr<ov::npuw::ICompiledModel> inner_compiled_model,
                                             std::unordered_map<std::string, size_t> dynamic_kv_cache_axes)
    : ov::npuw::ICompiledModel(outer_model, plugin),
      m_inner_compiled_model(std::move(inner_compiled_model)),
      m_dynamic_kv_cache_axes(std::move(dynamic_kv_cache_axes)) {
    OPENVINO_ASSERT(m_inner_compiled_model != nullptr, "GQACompiledModel requires a valid inner compiled model");
    m_dynamic_kv_cache_output_axes = find_dynamic_kv_cache_output_axes(outputs());
}

void ov::npuw::GQACompiledModel::write_port_list(std::ostream& stream,
                                                 const std::vector<ov::Output<const ov::Node>>& ports) {
    using namespace ov::npuw::s11n;
    write(stream, ports.size());
    for (const auto& port : ports) {
        write(stream, port.get_node()->get_friendly_name());
        write(stream, port.get_element_type().to_string());
        write(stream, port.get_partial_shape().to_string());
        // Tensor names (as opposed to the node's friendly name) are what ORT/OVEP use to
        // match its own input/output names to OpenVINO ports; losing them here silently
        // breaks that name-based lookup on the caller side even though our own
        // friendly-name-keyed m_dynamic_kv_cache_axes lookups would still work fine.
        write(stream, port.get_names());
    }
}

ov::ParameterVector ov::npuw::GQACompiledModel::read_input_port_list(std::istream& stream) {
    using namespace ov::npuw::s11n;
    size_t count = 0;
    read(stream, count);
    // Guards against misreading a mismatched/corrupted blob (e.g. an older-format blob
    // that slipped past the version check) as an absurd port count, which would
    // otherwise surface as an opaque bad_alloc/OOM instead of a clear error.
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
    for (const auto& [name, axis] : m_dynamic_kv_cache_axes) {
        GQA_TRACE("    '" << name << "' -> axis " << axis);
    }
    using namespace ov::npuw::s11n;
    write(stream, NPUW_SERIALIZATION_INDICATOR);
    write(stream, NPUW_GQA_COMPILED_MODEL_INDICATOR);
    write(stream, OPENVINO_VERSION_MAJOR);
    write(stream, OPENVINO_VERSION_MINOR);
    write(stream, OPENVINO_VERSION_PATCH);
    write(stream, std::string(NPUW_SERIALIZATION_VERSION));

    // Preserve the outer-facing ports (dynamic KV-cache/attention-bias axes as seen by
    // the caller) and the dynamic-axis map itself, so import_model() can rebuild an
    // equivalent GQACompiledModel wrapper instead of returning the bare inner
    // CompiledModel and losing the reshape/copy-prefix machinery on every cache hit.
    auto outer_inputs = inputs();
    auto outer_outputs = outputs();
    GQACompiledModel::write_port_list(stream, outer_inputs);
    GQACompiledModel::write_port_list(stream, outer_outputs);
    write(stream, m_dynamic_kv_cache_axes);
    GQA_TRACE("GQACompiledModel::export_model() -> wrote "
              << outer_inputs.size() << " outer inputs, " << outer_outputs.size() << " outer outputs, and "
              << m_dynamic_kv_cache_axes.size()
              << " dynamic-axis entries; delegating to inner m_inner_compiled_model->export_model()");
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

    ov::npuw::s11n::IndicatorType serialization_indicator;
    read(stream, serialization_indicator);
    NPUW_ASSERT(serialization_indicator == NPUW_SERIALIZATION_INDICATOR);

    ov::npuw::s11n::IndicatorType gqa_indicator;
    read(stream, gqa_indicator);
    NPUW_ASSERT(gqa_indicator == NPUW_GQA_COMPILED_MODEL_INDICATOR);

    int vmajor, vminor, vpatch;
    std::string s11n_version;
    read(stream, vmajor);
    read(stream, vminor);
    read(stream, vpatch);
    read(stream, s11n_version);

    GQA_TRACE("    blob version " << vmajor << '.' << vminor << '.' << vpatch << " / s11n '" << s11n_version
                                  << "', current is " << OPENVINO_VERSION_MAJOR << '.' << OPENVINO_VERSION_MINOR << '.'
                                  << OPENVINO_VERSION_PATCH << " / s11n '" << NPUW_SERIALIZATION_VERSION << "'");

    if (vmajor != OPENVINO_VERSION_MAJOR || vminor != OPENVINO_VERSION_MINOR || vpatch != OPENVINO_VERSION_PATCH ||
        s11n_version != std::string(NPUW_SERIALIZATION_VERSION)) {
        OPENVINO_THROW("GQA blob was serialized with a different OV version (",
                       vmajor,
                       '.',
                       vminor,
                       '.',
                       vpatch,
                       " / NPUW s11n ",
                       s11n_version,
                       "); current is ",
                       OPENVINO_VERSION_MAJOR,
                       '.',
                       OPENVINO_VERSION_MINOR,
                       '.',
                       OPENVINO_VERSION_PATCH,
                       " / NPUW s11n ",
                       NPUW_SERIALIZATION_VERSION);
    }

    // Rebuild the outer-facing model (with the dynamic KV-cache/attention-bias axes
    // restored to dynamic) and the axis map itself, exactly as they were before export.
    ov::ParameterVector outer_parameters;
    ov::NodeVector outer_results;
    std::unordered_map<std::string, size_t> dynamic_kv_cache_axes;
    outer_parameters = GQACompiledModel::read_input_port_list(stream);
    outer_results = GQACompiledModel::read_output_port_list(stream);
    read(stream, dynamic_kv_cache_axes);

    GQA_TRACE("    read " << outer_parameters.size() << " outer parameters, " << outer_results.size()
                          << " outer results, " << dynamic_kv_cache_axes.size() << " dynamic-axis entries:");
    for (const auto& [name, axis] : dynamic_kv_cache_axes) {
        GQA_TRACE("        '" << name << "' -> axis " << axis);
    }

    auto outer_model =
        std::make_shared<ov::Model>(ov::as_output_vector(outer_results), outer_parameters, "gqa_outer_model");

    // The rest of the stream is the inner CompiledModel ORC blob. It is fully
    // self-contained: partitioning is already baked in and port mappings are
    // consistent, positionally matching outer_parameters/outer_results above.
    auto inner_compiled_model = ov::npuw::CompiledModel::import_model(stream, plugin, properties);
    NPUW_ASSERT(inner_compiled_model != nullptr);
    GQA_TRACE("    inner CompiledModel imported @ "
              << inner_compiled_model.get()
              << "; wrapping it back into a GQACompiledModel so dynamic KV-cache/attention-bias reshape "
                 "survives the cache hit");

    return std::shared_ptr<GQACompiledModel>(
        new GQACompiledModel(outer_model, plugin, std::move(inner_compiled_model), std::move(dynamic_kv_cache_axes)));
}

std::shared_ptr<const ov::Model> ov::npuw::GQACompiledModel::get_runtime_model() const {
    return m_inner_compiled_model->get_runtime_model();
}

void ov::npuw::GQACompiledModel::set_property(const ov::AnyMap& properties) {
    m_inner_compiled_model->set_property(properties);
}

ov::Any ov::npuw::GQACompiledModel::get_property(const std::string& name) const {
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
    for (const auto& [name, axis] : m_compiled_model->m_dynamic_kv_cache_axes) {
        GQA_TRACE("    dynamic entry: name='" << name << "' axis=" << axis);
    }
}

void ov::npuw::GQAInferRequest::ensure_inner_request_locked() const {
    if (m_inner_request == nullptr) {
        GQA_TRACE("ensure_inner_request_locked(): creating inner infer request...");
        m_inner_request = m_compiled_model->m_inner_compiled_model->create_infer_request();
        OPENVINO_ASSERT(m_inner_request != nullptr, "GQA infer request requires a valid inner request");
        GQA_TRACE("ensure_inner_request_locked(): inner infer request created @ " << m_inner_request.get());
        const auto& inner_model = m_compiled_model->m_inner_compiled_model;
        GQA_TRACE("    inner compiled model inputs (" << inner_model->inputs().size() << "):");
        for (const auto& input : inner_model->inputs()) {
            GQA_TRACE("        '" << input.get_node()->get_friendly_name() << "' shape=" << input.get_partial_shape());
        }
        GQA_TRACE("    inner compiled model outputs (" << inner_model->outputs().size() << "):");
        for (const auto& output : inner_model->outputs()) {
            GQA_TRACE("        '" << output.get_node()->get_friendly_name()
                                  << "' shape=" << output.get_partial_shape());
        }

        // Zero-initialize the full physical capacity of every dynamic-axis KV-cache/bias
        // input (past_keys_N, past_values_N, attention_mask) right now, once.
        for (const auto& [name, axis] : m_compiled_model->m_dynamic_kv_cache_axes) {
            auto input_it = std::find_if(inner_model->inputs().begin(),
                                         inner_model->inputs().end(),
                                         [&name](const ov::Output<const ov::Node>& input) {
                                             return input.get_node()->get_friendly_name() == name;
                                         });
            if (input_it == inner_model->inputs().end()) {
                GQA_TRACE("    zero-init: '" << name << "' not found among inner inputs, skipping");
                continue;
            }
            auto tensor = m_inner_request->get_tensor(*input_it);
            if (!tensor || tensor->get_byte_size() == 0) {
                continue;
            }
            std::memset(tensor->data(), 0, tensor->get_byte_size());
            GQA_TRACE("    zero-init: '" << name << "' axis=" << axis << " shape=" << tensor->get_shape() << " ("
                                         << tensor->get_byte_size() << " bytes) zeroed");
        }
    }
}

// FIXME: This is an online search every time. The data itself isn't dynamic,
// a static mapping can be created once and used for all queries.
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

// Reads a scalar (or single-element) integer tensor's value, regardless of whether it
// was stored as i32 or i64 -- this is how GQA's various length/position inputs
// (seqlens_k, past_seq_len, total_seq_len, ...) can be typed by different producers.
static std::optional<int64_t> read_scalar_int_tensor(const ov::SoPtr<ov::ITensor>& tensor) {
    if (!tensor || ov::shape_size(tensor->get_shape()) == 0) {
        return std::nullopt;
    }
    switch (tensor->get_element_type()) {
    case ov::element::i32:
        return static_cast<int64_t>(*tensor->data<int32_t>());
    case ov::element::i64:
        return *tensor->data<int64_t>();
    default:
        return std::nullopt;
    }
}

// Diagnostic-only: dumps the current values of every known "sequence length" style
// input this model happens to expose, read straight from the *inner* request's tensor
// right before infer() -- i.e. after any writes the caller may have done directly into
// a get_tensor()'d buffer (not just via set_tensor()), so this reflects exactly what
// the NPU is about to compute against. Names are probed rather than assumed, since
// different ONNX exporters/deployments name (or omit) these inputs differently.
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
        std::ostringstream value_str;
        if (auto value = read_scalar_int_tensor(tensor)) {
            value_str << *value;
        } else {
            value_str << "<unreadable: element_type=" << tensor->get_element_type()
                      << ", size=" << ov::shape_size(tensor->get_shape()) << ">";
        }
        GQA_TRACE("pre-infer sequence-length input '" << candidate << "' shape=" << tensor->get_shape()
                                                      << " value=" << value_str.str());
    }
}

// Diagnostic-only: computes {min, max, count of non-zero elements, total element count}
// over a tensor's numeric content, regardless of fp16/fp32/int dtype. Used to answer
// "does this model's attention_mask/bias input actually carry real values, or is it
// permanently zero/inert (as it is in a comparable reference deployment, which never
// writes to it in its single-token decode fast path and relies entirely on GQA's own
// internal past_seqlen-driven causal mask)?"
namespace {
struct TensorStats {
    double min_value = 0.0;
    double max_value = 0.0;
    size_t non_zero_count = 0;
    size_t total_count = 0;
};

template <typename T>
TensorStats compute_stats_typed(const T* data, size_t count) {
    TensorStats stats;
    stats.total_count = count;
    if (count == 0) {
        return stats;
    }
    double min_value = static_cast<double>(data[0]);
    double max_value = static_cast<double>(data[0]);
    size_t non_zero_count = 0;
    for (size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(data[i]);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        if (value != 0.0) {
            ++non_zero_count;
        }
    }
    stats.min_value = min_value;
    stats.max_value = max_value;
    stats.non_zero_count = non_zero_count;
    return stats;
}

std::optional<TensorStats> compute_tensor_stats(const ov::SoPtr<ov::ITensor>& tensor) {
    if (!tensor) {
        return std::nullopt;
    }
    const auto count = ov::shape_size(tensor->get_shape());
    switch (tensor->get_element_type()) {
    case ov::element::f32:
        return compute_stats_typed(tensor->data<float>(), count);
    case ov::element::f16:
        return compute_stats_typed(tensor->data<ov::float16>(), count);
    case ov::element::i32:
        return compute_stats_typed(tensor->data<int32_t>(), count);
    case ov::element::i64:
        return compute_stats_typed(tensor->data<int64_t>(), count);
    default:
        return std::nullopt;
    }
}
}  // namespace

// Diagnostic-only: dumps min/max/non-zero-count for every dynamic-axis input that isn't
// a past_key/past_value KV-cache tensor -- i.e. the attention bias/mask input(s) located
// via find_dynamic_kv_cache_axes()'s ATTENTION_BIAS lookup. Read straight from the inner
// request's tensor right before infer(), after this call's set_tensor() has already
// copied the caller's data into it.
void ov::npuw::GQAInferRequest::trace_attention_mask_stats_locked() const {
    if (!gqa_trace_enabled()) {
        return;
    }
    const auto& outer_inputs = m_compiled_model->inputs();
    for (const auto& [name, axis] : m_compiled_model->m_dynamic_kv_cache_axes) {
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
        std::ostringstream stats_str;
        if (auto stats = compute_tensor_stats(tensor)) {
            stats_str << " min=" << stats->min_value << " max=" << stats->max_value
                      << " non_zero=" << stats->non_zero_count << "/" << stats->total_count;
        } else {
            stats_str << " <stats unavailable for this element_type>";
        }
        GQA_TRACE("pre-infer attention-bias/mask input '"
                  << name << "' axis=" << axis << " shape=" << tensor->get_shape()
                  << " element_type=" << tensor->get_element_type() << stats_str.str());
    }
}

void ov::npuw::GQAInferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    sync_dynamic_kv_cache_tensors_locked();
    trace_sequence_length_inputs_locked();
    trace_attention_mask_stats_locked();
    GQA_TRACE("GQAInferRequest::infer() -> delegating to inner request @ " << m_inner_request.get());
    // TODO: It is probably better to enclose the code with try/catch only if necessary,
    // e.g. when tracing is enabled. A generic snippet can be helpful here
    try {
        m_inner_request->infer();
    } catch (const std::exception& ex) {
        // The app-level error codes (e.g. PsResult::InferenceError) surfaced to the user
        // carry no detail about what actually failed inside OV/the NPU driver -- print the
        // real exception message here before it propagates and gets wrapped away.
        GQA_TRACE("GQAInferRequest::infer() -> inner request infer() THREW: " << ex.what());
        throw;
    } catch (...) {
        GQA_TRACE("GQAInferRequest::infer() -> inner request infer() THREW an unknown exception");
        throw;
    }
    GQA_TRACE("GQAInferRequest::infer() -> inner request infer() returned");
    refresh_present_tensors_locked();
}

// Refreshes the outer-facing tensors for dynamic present_key/present_value outputs
// right after infer() completes. The inner request's own tensor for such an output is
// the full-capacity static buffer (e.g. 2048) with new K/V scattered in-place at
// [past_seq_len, past_seq_len + new_seq_len); anything beyond that is stale data from a
// previous step, not something the caller should see. When trimming is actually needed
// (valid_len < capacity), this copies the valid prefix into a densely-shaped tensor --
// cached per output name in m_dynamic_kv_cache_output_tensors and reused/overwritten
// call-over-call, only (re)allocated when a private buffer isn't already there or the
// shape changed -- that get_tensor() then hands back as-is. When no trimming is needed
// (valid_len == capacity), the outer tensor is aliased directly onto the inner tensor
// instead, skipping the copy entirely (see m_dynamic_kv_cache_output_aliased).
void ov::npuw::GQAInferRequest::refresh_present_tensors_locked() const {
    const auto& output_axes = m_compiled_model->m_dynamic_kv_cache_output_axes;
    if (output_axes.empty()) {
        return;
    }

    const auto& outer_outputs = m_compiled_model->outputs();
    for (const auto& [name, axis] : output_axes) {
        auto port_it = std::find_if(outer_outputs.begin(), outer_outputs.end(), [&](const auto& output) {
            return output.get_node()->get_friendly_name() == name;
        });
        if (port_it == outer_outputs.end()) {
            continue;
        }
        const auto inner_tensor = m_inner_request->get_tensor(map_port_locked(*port_it));
        const auto capacity = inner_tensor->get_shape().at(axis);

        // NB: Poor mans' output shape inference here.
        // Resolve this output's shape length: mirror the corresponding past_key/value
        // input's own set length verbatim; otherwise fall back to the full buffer.
        size_t valid_len = capacity;
        if (auto past_name = ov::npuw::GQACompiledModel::present_to_past_name(name)) {
            auto past_it = m_dynamic_kv_cache_tensors.find(*past_name);
            if (past_it != m_dynamic_kv_cache_tensors.end()) {
                valid_len = std::min<size_t>(capacity, past_it->second->get_shape().at(axis));
            }
        }

        if (valid_len == capacity) {
            // No trimming needed -- the whole inner buffer is valid, so alias the outer
            // tensor directly onto it instead of copying. NEVER call set_shape() on this
            // entry while it's aliased (see m_dynamic_kv_cache_output_aliased); it's the
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
            // Either never allocated, or the existing entry is an alias onto the inner
            // tensor from a previous no-trimming call -- can't resize that in place
            // (see above), so allocate a fresh, privately-owned buffer instead.
            // FIXME: CPU allocation here
            outer_tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(inner_tensor->get_element_type(), out_shape));
        } else if (outer_tensor->get_shape() != out_shape) {
            // Resize in place -- keeps this tensor's object identity stable across calls
            // (matters if the caller/ORT-OVEP IO binding cached the returned object or
            // its data pointer), rather than handing back a different tensor instance.
            outer_tensor->set_shape(out_shape);
        }

        // A strided ROI view over the inner buffer's [0, valid_len) prefix, purely as a
        // vehicle for copy_kv_cache_prefix's stride-aware per-plane copy below -- never
        // handed to the caller directly (its shape looks dense but its memory isn't).
        ov::Coordinate begin(inner_tensor->get_shape().size(), 0);
        ov::Coordinate end(inner_tensor->get_shape());
        end.at(axis) = valid_len;
        auto inner_view = ov::SoPtr<ov::ITensor>(ov::make_tensor(inner_tensor._ptr, begin, end));
        ov::npuw::GQACompiledModel::copy_kv_cache_prefix(inner_view, outer_tensor, axis);
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::GQAInferRequest::get_present_tensor_locked(const std::string& name,
                                                                            size_t axis) const {
    (void)axis;
    auto it = m_dynamic_kv_cache_output_tensors.find(name);
    if (it != m_dynamic_kv_cache_output_tensors.end()) {
        return it->second;
    }
    // infer() hasn't run (or refreshed this port) yet -- ORT/OVEP may legitimately query
    // an output tensor before the first infer() (e.g. at output-binding setup time).
    // Fall back to the inner request's own (raw, static-capacity) tensor rather than
    // hard-failing a call that used to succeed before this trimming was added.
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
        // The inner request's tensor for this port is the static, configured-capacity
        // buffer -- not what the caller set. Hand back the exact user-owned tensor
        // instead; there is nothing to allocate on our side for the dynamic shape.
        auto it = m_dynamic_kv_cache_tensors.find(name);
        OPENVINO_ASSERT(it != m_dynamic_kv_cache_tensors.end(),
                        "GQA KV-cache '",
                        name,
                        "' has a dynamic max_seq_len; set_tensor() must be called before get_tensor()");
        GQA_TRACE("    -> returning previously set_tensor()'d user tensor, shape=" << it->second->get_shape());
        return it->second;
    }

    if (auto out_it = m_compiled_model->m_dynamic_kv_cache_output_axes.find(name);
        out_it != m_compiled_model->m_dynamic_kv_cache_output_axes.end()) {
        auto tensor = get_present_tensor_locked(name, out_it->second);
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

    const auto& dynamic_axes = m_compiled_model->m_dynamic_kv_cache_axes;
    auto it = dynamic_axes.find(name);
    if (it == dynamic_axes.end()) {
        GQA_TRACE("    -> not a dynamic-axis input, forwarding to inner request as-is");
        m_inner_request->set_tensor(map_port_locked(port), tensor);
        return;
    }

    // This KV-cache input was compiled with a static capacity for NPU compilation. If the
    // caller-supplied tensor is EXACTLY that capacity (not just <=), there's no "valid
    // prefix" to carve out -- the whole buffer is meant to be seen, so we can alias the
    // inner request directly onto the caller's tensor and skip the prefix copy entirely
    // (this is also the largest, most expensive copy case, since it's a full-capacity
    // buffer). Otherwise, keep the inner request's own (already-allocated, statically-
    // shaped) tensor and defer copying the variable-length prefix to infer() time.
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
        // Exact match: alias the inner request directly onto the caller's tensor instead
        // of copying into a separate, static buffer -- no copy is needed since the whole
        // buffer is valid and there's nothing to trim.
        m_inner_request->set_tensor(map_port_locked(port), tensor);
        m_dynamic_kv_cache_tensors[name] = tensor;
        GQA_TRACE("    -> requested_len == capacity, aliased inner request tensor directly (no copy)");
        return;
    }
    // Data is intentionally NOT copied here: the caller may keep writing into `tensor`
    // after this call and before infer() (e.g. set_tensor(t); write_into(t); infer();).
    // Only sync_dynamic_kv_cache_tensors_locked(), called right before the inner
    // request's own infer(), is guaranteed to see the tensor's live content.
    m_dynamic_kv_cache_tensors[name] = tensor;
    GQA_TRACE("    -> stored user tensor reference for later sync at infer() time");
}

void ov::npuw::GQAInferRequest::sync_dynamic_kv_cache_tensors_locked() const {
    const auto& outer_inputs = m_compiled_model->inputs();
    for (const auto& [name, axis] : m_compiled_model->m_dynamic_kv_cache_axes) {
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
            // set_tensor() already aliased the inner request directly onto this tensor
            // (the exact-capacity case) -- it's the very same object, so there's nothing
            // to copy; doing so would just be a wasteful full-buffer self-copy.
            GQA_TRACE("sync_dynamic_kv_cache_tensors_locked(): '" << name << "' already aliased, skipping copy");
            continue;
        }
        GQA_TRACE("sync_dynamic_kv_cache_tensors_locked(): '"
                  << name << "' axis=" << axis << " copying live prefix, shape=" << tensor_it->second->get_shape());
        ov::npuw::GQACompiledModel::copy_kv_cache_prefix(tensor_it->second, inner_tensor, axis);
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
