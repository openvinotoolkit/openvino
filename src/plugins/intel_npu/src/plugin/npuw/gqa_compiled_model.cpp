// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa_compiled_model.hpp"

#include <utility>

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

    ov::AnyMap config = {
        {"NPUW_ONLINE_PIPELINE", "REP"},
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
    return {model, std::move(prepared_properties)};
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

ov::npuw::GQACompiledModel::GQACompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::AnyMap& properties,
                                             CompiledModelFactory factory)
    : GQACompiledModel(prepare(model, properties), plugin, std::move(factory)) {}

ov::npuw::GQACompiledModel::GQACompiledModel(PreparedState prepared,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             CompiledModelFactory factory)
    : ov::npuw::ICompiledModel(prepared.model, plugin),
      m_compiled_model(factory(prepared.model, plugin, prepared.properties)) {
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
    return m_inner_request->get_tensor(map_port_locked(port));
}

void ov::npuw::GQAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                           const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    m_inner_request->set_tensor(map_port_locked(port), tensor);
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
