// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa_compiled_model.hpp"

#include <optional>
#include <utility>

#include "infer_request_utils.hpp"
#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "npuw_transformations/collapse_unqdq.hpp"
#include "npuw_transformations/conv_to_matmul.hpp"
#include "npuw_transformations/drop_zp_subtract.hpp"
#include "npuw_transformations/untangle_dq_scale.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "util.hpp"

namespace {

// Static capacity (in tokens) a dynamic KV-cache is reshaped to for NPU compilation.
constexpr size_t kMaxSeqLen = 32768;

// Scans `model`'s past_key/past_value Parameters for a dynamic dimension and returns the
// axis to pin to kMaxSeqLen, keyed by Parameter friendly name. A KV-cache Parameter with
// more than one dynamic dimension is ambiguous and not something we can safely resolve.
std::unordered_map<std::string, size_t> find_dynamic_kv_cache_axes(const std::shared_ptr<const ov::Model>& model) {
    std::unordered_map<std::string, size_t> result;
    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (!ov::npuw::util::contains_ignore_case(name, "past_key") &&
            !ov::npuw::util::contains_ignore_case(name, "past_value")) {
            continue;
        }
        const auto& partial_shape = parameter->get_partial_shape();
        if (partial_shape.rank().is_dynamic()) {
            continue;
        }
        std::optional<size_t> dynamic_axis;
        for (size_t i = 0; i < partial_shape.size(); ++i) {
            if (partial_shape[i].is_dynamic()) {
                OPENVINO_ASSERT(!dynamic_axis.has_value(),
                                "GQA KV-cache parameter '",
                                name,
                                "' has more than one dynamic dimension; can't resolve its max_seq_len axis");
                dynamic_axis = i;
            }
        }
        if (dynamic_axis.has_value()) {
            result.emplace(name, *dynamic_axis);
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

    // Disable partitioning in V1 case.
    const auto gqa_case = ov::npuw::GQACompiledModel::identify_case(model);
    const char* online_pipeline = gqa_case == ov::npuw::GQACompiledModel::Case::V1 ? "NONE" : "REP";

    ov::AnyMap config = {
        {"NPUW_ONLINE_PIPELINE", online_pipeline},
        {std::string(::intel_npu::NPUW_DEVICES::key()), "NPU"},
        {ov::cache_mode.name(), ov::CacheMode::OPTIMIZE_SPEED},
        {std::string(::intel_npu::NPUW_UNQDQ::key()), "YES"},
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
        LOG_INFO("Detected generate-style GQA model; applying FOLD with async funcall");
    } else {
        LOG_INFO("GQA model stage unknown; FOLD disabled");
    }
    merge_config_with(config, properties);
    return {config, stage};
}

}  // namespace

ov::npuw::GQACompiledModel::PreparedState ov::npuw::GQACompiledModel::prepare(const std::shared_ptr<ov::Model>& model,
                                                                              const ov::AnyMap& properties) {
    auto [prepared_properties, stage] = with_gqa_defaults(model, properties);

    model->set_friendly_name(model->get_friendly_name() + "_gqa_" +
                             (stage == GQAModelStage::PREFILL    ? "prefill"
                              : stage == GQAModelStage::GENERATE ? "generate"
                                                                 : "unknown"));

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
    std::unordered_map<std::string, size_t> dynamic_kv_cache_axis;
    if (has_dynamic_max_seq_len(model)) {
        dynamic_kv_cache_axis = find_dynamic_kv_cache_axes(model);
        OPENVINO_ASSERT(!dynamic_kv_cache_axis.empty(),
                        "GQA model has a dynamic max_seq_len but no resolvable KV-cache Parameter was found");
        compiled_model = model->clone();
        std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
        for (const auto& [name, axis] : dynamic_kv_cache_axis) {
            const auto& params = compiled_model->get_parameters();
            auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
                return parameter->get_friendly_name() == name;
            });
            OPENVINO_ASSERT(it != params.end(), "KV-cache parameter '", name, "' not found in the cloned model");
            auto new_shape = (*it)->get_partial_shape();
            new_shape[axis] = ov::Dimension(static_cast<int64_t>(kMaxSeqLen));
            new_shapes[(*it)->output(0)] = new_shape;
        }
        compiled_model->reshape(new_shapes);
        for (const auto& [name, axis] : dynamic_kv_cache_axis) {
            const auto& params = compiled_model->get_parameters();
            auto it = std::find_if(params.begin(), params.end(), [&](const auto& parameter) {
                return parameter->get_friendly_name() == name;
            });
            OPENVINO_ASSERT(it != params.end() && (*it)->get_partial_shape().is_static(),
                            "Reshaping the GQA KV-cache parameter '",
                            name,
                            "' to a static capacity of ",
                            kMaxSeqLen,
                            " did not make it fully static");
        }
        LOG_INFO("Reshaped dynamic GQA KV-cache to a static capacity of " << kMaxSeqLen << " tokens");
    }

    return {model, compiled_model, std::move(prepared_properties), std::move(dynamic_kv_cache_axis)};
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
    return identify_case(model) != Case::Unknown;
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
    }
    return false;
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
    // chunk; copy every row's S1 prefix into the wider S2-capacity row, left-aligned. This
    // is the axis==3 analog of copy_by_planes, walking the same N=1/H/E "rows".
    OPENVINO_ASSERT(src_shape[0] == 1u, "Expected batch size 1");
    OPENVINO_ASSERT(src_shape[0] == dst_shape[0] && src_shape[1] == dst_shape[1] && src_shape[2] == dst_shape[2]);
    const auto* src_p = reinterpret_cast<const uint8_t*>(src->data());
    auto* dst_p = reinterpret_cast<uint8_t*>(dst->data());
    const auto rows = src_shape[1] * src_shape[2];
    const auto src_row_stride = src->get_strides()[2];
    const auto dst_row_stride = dst->get_strides()[2];
    const auto row_bytes = src_row_stride;  // src's own S1-sized contiguous row
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
      m_compiled_model(factory(prepared.compiled_model, plugin, prepared.properties)),
      m_dynamic_kv_cache_axis(std::move(prepared.dynamic_kv_cache_axis)) {
    OPENVINO_ASSERT(m_compiled_model != nullptr, "GQACompiledModel requires a valid inner compiled model");
}

void ov::npuw::GQACompiledModel::export_model(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, NPUW_SERIALIZATION_INDICATOR);
    write(stream, NPUW_GQA_COMPILED_MODEL_INDICATOR);
    write(stream, OPENVINO_VERSION_MAJOR);
    write(stream, OPENVINO_VERSION_MINOR);
    write(stream, OPENVINO_VERSION_PATCH);
    write(stream, std::string(NPUW_SERIALIZATION_VERSION));
    m_compiled_model->export_model(stream);
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::GQACompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_INFO("Deserializing GQACompiledModel...");
    LOG_BLOCK();

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

    // The rest of the stream is the inner CompiledModel ORC blob.
    // After import it is fully self-contained; no outer GQA wrapper is needed
    // because the partitioning is already baked in and port mappings are consistent.
    return ov::npuw::CompiledModel::import_model(stream, plugin, properties);
}

std::shared_ptr<const ov::Model> ov::npuw::GQACompiledModel::get_runtime_model() const {
    return m_compiled_model->get_runtime_model();
}

void ov::npuw::GQACompiledModel::set_property(const ov::AnyMap& properties) {
    m_compiled_model->set_property(properties);
}

ov::Any ov::npuw::GQACompiledModel::get_property(const std::string& name) const {
    return m_compiled_model->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::GQACompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const GQACompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::GQAInferRequest>(std::move(self));
}

ov::npuw::GQAInferRequest::GQAInferRequest(std::shared_ptr<const GQACompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(std::move(compiled_model)) {}

void ov::npuw::GQAInferRequest::ensure_inner_request_locked() const {
    if (m_inner_request == nullptr) {
        m_inner_request = m_compiled_model->m_compiled_model->create_infer_request();
        OPENVINO_ASSERT(m_inner_request != nullptr, "GQA infer request requires a valid inner request");
    }
}

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

void ov::npuw::GQAInferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    m_inner_request->infer();
}

ov::SoPtr<ov::ITensor> ov::npuw::GQAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    const auto& name = port.get_node()->get_friendly_name();
    if (m_compiled_model->m_dynamic_kv_cache_axis.count(name) != 0) {
        // The inner request's tensor for this port is the static, kMaxSeqLen-sized
        // buffer -- not what the caller set. Hand back the exact user-owned tensor
        // instead; there is nothing to allocate on our side for the dynamic shape.
        auto it = m_dynamic_kv_cache_tensors.find(name);
        OPENVINO_ASSERT(it != m_dynamic_kv_cache_tensors.end(),
                        "GQA KV-cache '",
                        name,
                        "' has a dynamic max_seq_len; set_tensor() must be called before get_tensor()");
        return it->second;
    }

    return m_inner_request->get_tensor(map_port_locked(port));
}

void ov::npuw::GQAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                           const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    const auto& name = port.get_node()->get_friendly_name();
    const auto& dynamic_axes = m_compiled_model->m_dynamic_kv_cache_axis;
    auto it = dynamic_axes.find(name);
    if (it == dynamic_axes.end()) {
        m_inner_request->set_tensor(map_port_locked(port), tensor);
        return;
    }

    // This KV-cache input was compiled with a static capacity of kMaxSeqLen: keep the
    // inner request's own (already-allocated, statically-shaped) tensor and copy the
    // user-supplied, variable-length data into its valid prefix instead of replacing it.
    const auto axis = it->second;
    const auto& inner_tensor = m_inner_request->get_tensor(map_port_locked(port));
    const auto requested_len = tensor->get_shape().at(axis);
    const auto capacity = inner_tensor->get_shape().at(axis);
    OPENVINO_ASSERT(requested_len <= capacity,
                    "GQA KV-cache '",
                    name,
                    "' length ",
                    requested_len,
                    " exceeds the static capacity (",
                    capacity,
                    ") it was compiled with");
    ov::npuw::GQACompiledModel::copy_kv_cache_prefix(tensor, inner_tensor, axis);
    m_dynamic_kv_cache_tensors[name] = tensor;
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
