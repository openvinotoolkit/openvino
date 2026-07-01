// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gqa_compiled_model.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <utility>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "npuw_transformations/collapse_unqdq.hpp"
#include "npuw_transformations/conv_to_matmul.hpp"
#include "npuw_transformations/drop_zp_subtract.hpp"
#include "npuw_transformations/untangle_dq_scale.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

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

ov::AnyMap with_gqa_defaults(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& properties) {
    enum class GQAModelStage {
        UNKNOWN,
        PREFILL,
        GENERATE,
    };

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

    const auto gqa_managed_key = std::string(::intel_npu::NPUW_GQA_MANAGED::key());
    const bool gqa_managed = properties.count(gqa_managed_key) && properties.at(gqa_managed_key).as<bool>();
    LOG_INFO("GQACompiledModel: gqa_managed=" << gqa_managed << " (key_present=" << properties.count(gqa_managed_key)
                                              << ")");

    const auto stage = detect_gqa_model_stage();
    if (stage == GQAModelStage::PREFILL) {
        if (gqa_managed) {
            // GQA decomposition uses ScatterUpdate in the KV path — the standard ATTN isolation
            // pattern (which expects Concat) will not match.  Apply FOLD without ATTN isolation
            // and let FOLD find the repeating [ffn+attn] blocks directly.
            merge_config_with(config, {{std::string(::intel_npu::NPUW_FOLD::key()), "YES"}});
            LOG_INFO("Detected prefill-style GQA model (managed); applying FOLD without ATTN isolation");
        } else {
            merge_config_with(config,
                              {{std::string(::intel_npu::NPUW_FOLD::key()), "YES"},
                               {"NPUW_FOLD_ONLY", "attn"},
                               {"NPUW_ONLINE_ISOLATE", "ATTN"},
                               {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "2"},  // Avoid applying attention policies here
                               {"NPUW_ATTN", "STATIC"}});
            LOG_INFO("Detected prefill-style GQA model; applying FOLD with ATTN isolation");
        }
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
    return config;
}

}  // namespace

// Build a minimal stub model that advertises the right input/output shapes for
// the user-facing (outer) GQA interface.  KV outputs have max_seq in the sequence
// dimension (dim[2] for K, dim[3] for transposed V); everything else matches the
// inner compiled model exactly.
// Used when reconstructing a GQACompiledModel from an imported blob.
//
// NOTE: stub Parameters (one per output) are added to the model's ParameterVector
// AFTER the real input parameters to satisfy ov::Model::check_all_parameters_registered.
// They are hidden from the user-facing interface by GQACompiledModel::inputs().
static std::shared_ptr<ov::Model> build_stub_outer_model(const std::shared_ptr<ov::npuw::ICompiledModel>& inner_cm,
                                                         const std::vector<size_t>& kv_indices,
                                                         const std::vector<size_t>& kv_max_seqs,
                                                         const std::vector<bool>& kv_transposed) {
    ov::ParameterVector input_params;
    for (const auto& inp : inner_cm->inputs()) {
        auto p = std::make_shared<ov::op::v0::Parameter>(inp.get_element_type(), inp.get_partial_shape());
        p->set_friendly_name(inp.get_node()->get_friendly_name());
        // OVEP matches inputs by tensor name, not friendly name.
        // ICompiledModel doesn't populate tensor names, so use friendly name as tensor name.
        const auto& tnames = inp.get_tensor().get_names();
        if (!tnames.empty())
            p->get_output_tensor(0).set_names(tnames);
        else
            p->get_output_tensor(0).set_names({inp.get_node()->get_friendly_name()});
        input_params.push_back(p);
    }

    ov::ParameterVector stub_params;
    ov::ResultVector results;
    size_t kv_i = 0;
    for (size_t i = 0; i < inner_cm->outputs().size(); ++i) {
        const auto& out = inner_cm->outputs()[i];
        bool is_kv = std::find(kv_indices.begin(), kv_indices.end(), i) != kv_indices.end();
        ov::PartialShape shape = out.get_partial_shape();
        if (is_kv && kv_i < kv_max_seqs.size()) {
            if (kv_i < kv_transposed.size() && kv_transposed[kv_i]) {
                // Transposed V: inner shape is [1, kv_heads, curr_seq, head_size],
                // outer (user-facing) shape must be [1, kv_heads, head_size, max_seq].
                ov::Dimension kv_heads = shape[1];
                ov::Dimension head_size = shape[3];
                shape = ov::PartialShape{shape[0], kv_heads, head_size, ov::Dimension(kv_max_seqs[kv_i])};
            } else {
                // Standard K/V: inner shape [1, kv_heads, curr_seq, head_size],
                // outer shape [1, kv_heads, max_seq, head_size].
                shape[2] = kv_max_seqs[kv_i];
            }
            ++kv_i;
        }
        // Stub parameter feeds this result; it is NOT a real model input but must be
        // registered in the model's ParameterVector to pass check_all_parameters_registered.
        auto stub = std::make_shared<ov::op::v0::Parameter>(out.get_element_type(), shape);
        stub->set_friendly_name("__gqa_out_stub_" + std::to_string(i));
        stub_params.push_back(stub);
        auto r = std::make_shared<ov::op::v0::Result>(stub);
        r->set_friendly_name(out.get_node()->get_friendly_name());
        // Set tensor name on the stub's output so OVEP can match outputs by name.
        const auto& otnames = out.get_tensor().get_names();
        if (!otnames.empty())
            stub->get_output_tensor(0).set_names(otnames);
        else
            stub->get_output_tensor(0).set_names({out.get_node()->get_friendly_name()});
        results.push_back(r);
    }

    // All params = real inputs first, then stubs.  Stubs are appended so that
    // GQACompiledModel::inputs() can trivially slice them off by knowing real_input_count.
    ov::ParameterVector all_params = input_params;
    all_params.insert(all_params.end(), stub_params.begin(), stub_params.end());
    return std::make_shared<ov::Model>(results, all_params, "gqa_managed_outer");
}

ov::npuw::GQACompiledModel::PreparedState ov::npuw::GQACompiledModel::prepare(const std::shared_ptr<ov::Model>& model,
                                                                              const ov::AnyMap& properties) {
    auto prepared_properties = with_gqa_defaults(model, properties);
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
    // Apply the core GQA decomposition (ScatterUpdate-based left-aligned KV cache) when requested.
    const auto gqa_managed_key = std::string(::intel_npu::NPUW_GQA_MANAGED::key());
    if (prepared_properties.count(gqa_managed_key) && prepared_properties.at(gqa_managed_key).as<bool>()) {
        // Safety valve: GQA_MANAGED relies on the static ScatterUpdate KV path from
        // GroupQueryAttentionDecomposition.  With slice_kv the model uses dynamic
        // shapes (past_key dim[2] is dynamic) and decomposition produces Concat instead
        // of ScatterUpdate — GQA_MANAGED cannot work in that case.
        bool has_slice_kv = false;
        for (const auto& op : model->get_ops()) {
            auto gqa = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(op);
            if (!gqa || gqa->get_input_size() <= 3)
                continue;
            const auto& ps = gqa->input_value(3).get_partial_shape();  // past_key
            if (ps.rank().is_dynamic() || ps[2].is_dynamic()) {
                has_slice_kv = true;
                break;
            }
        }
        if (has_slice_kv) {
            LOG_WARN("NPUW_GQA_MANAGED: slice_kv model detected (dynamic past_key dim[2]) -- "
                     "GQA_MANAGED requires static KV shapes; disabling");
            return {model, nullptr, std::move(prepared_properties), false, {}, {}, {}};
        }
        // Identify the seqlens_k parameter BEFORE decomposition by reading input 5 of the
        // first GroupQueryAttention op.  After decomposition the GQA node is gone and input 5
        // becomes the seqlens scalar that feeds the scatter-index Range computation.
        // Input 6 (total_sequence_length) is read by the GQA op but unused in the decomposed
        // subgraph — detecting it by name substring would be wrong.
        std::string seqlens_k_name;
        for (const auto& op : model->get_ops()) {
            auto gqa = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(op);
            if (!gqa || gqa->get_input_size() <= 5)
                continue;
            auto param = ov::as_type_ptr<ov::op::v0::Parameter>(gqa->input_value(5).get_node_shared_ptr());
            if (param) {
                seqlens_k_name = param->get_friendly_name();
                LOG_INFO("NPUW_GQA_MANAGED: detected seqlens_k parameter '" << seqlens_k_name << "'");
            }
            break;  // all layers share the same seqlens_k
        }
        if (seqlens_k_name.empty()) {
            LOG_INFO("NPUW_GQA_MANAGED: seqlens_k not found in any GQA op input 5; integer parameters:");
            for (const auto& p : model->get_parameters()) {
                const auto& et = p->get_element_type();
                if (et == ov::element::i32 || et == ov::element::i64)
                    LOG_INFO("  " << p->get_friendly_name() << " " << et << " " << p->get_partial_shape());
            }
        }

        LOG_INFO("NPUW_GQA_MANAGED: applying GroupQueryAttentionDecomposition");
        ov::pass::GraphRewrite rewrite;
        rewrite.add_matcher<ov::pass::GroupQueryAttentionDecomposition>();
        rewrite.run_on_model(model);

        // Clone the decomposed model to preserve full-shape KV outputs for the outer interface.
        auto outer_model = model->clone();

        // Strip ScatterUpdate from KV result outputs on the inner model so the NPU only
        // produces the new token slice (not the full KV cache) on each step.
        // Also handles transposed V: Result ← Transpose ← ScatterUpdate.
        // In that case both Transpose and ScatterUpdate are stripped so the inner output
        // is the raw V slice [1, kv_heads, curr_seq, head_size], and the CPU scatter
        // copies it into the user's transposed [1, kv_heads, head_size, max_seq] tensor.
        std::vector<size_t> kv_indices;
        std::vector<bool> kv_transposed_flags;
        const auto& results = model->get_results();
        for (size_t i = 0; i < results.size(); ++i) {
            auto direct_src = results[i]->input_value(0).get_node_shared_ptr();
            auto scatter = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(direct_src);
            bool transposed = false;
            if (!scatter) {
                // Check for Transpose ← ScatterUpdate pattern.
                auto trans = ov::as_type_ptr<ov::op::v1::Transpose>(direct_src);
                if (trans) {
                    scatter = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(trans->input_value(0).get_node_shared_ptr());
                    if (scatter)
                        transposed = true;
                }
            }
            if (scatter) {
                // The GQA decomposition always scatters at axis=2 which is the sequence
                // dimension for standard layout {B,H,seq,head}. When the model uses transposed
                // V cache {B,H,head_size,max_seq}, the app wraps the ScatterUpdate output with
                // a Transpose op — the 'transposed' flag captures this pattern.
                // In that case axis=2 operates on head_size (wrong dim), so CPU-managed scatter
                // is not applicable. Leave this output untouched so the NPU handles it natively.
                if (transposed) {
                    LOG_INFO("NPUW_GQA_MANAGED: output[" << i
                                                         << "] has Transpose wrapper on ScatterUpdate"
                                                            "; skipping CPU scatter (NPU handles natively)");
                    continue;  // leave Result → Transpose → ScatterUpdate intact in the inner model
                }
                // Redirect result to carry only the current-token slice (scatter's "updates").
                results[i]->input(0).replace_source_output(scatter->input_value(2));
                kv_indices.push_back(i);
                kv_transposed_flags.push_back(transposed);
            }
        }

        // After redirecting K Results to curr_k, re-run shape inference so that NPUW sees the
        // correct (small) output shapes instead of the stale ScatterUpdate shapes.
        if (!kv_indices.empty()) {
            model->validate_nodes_and_infer_types();
        }

        // Capture KV max_seq values from the outer (pre-strip) model.
        // All managed outputs use standard layout — sequence dim is [2].
        std::vector<size_t> kv_max_seqs;
        const auto& outer_results = outer_model->get_results();
        for (size_t ki = 0; ki < kv_indices.size(); ++ki) {
            const auto& ps = outer_results[kv_indices[ki]]->input_value(0).get_partial_shape();
            const size_t seq_dim = kv_transposed_flags[ki] ? 3 : 2;
            kv_max_seqs.push_back(ps[seq_dim].get_length());
        }

        const bool can_manage_kv = !kv_indices.empty() && !seqlens_k_name.empty();
        LOG_INFO("NPUW_GQA_MANAGED: stripped "
                 << kv_indices.size() << " KV output(s)"
                 << (can_manage_kv ? "; infer request will scatter into user KV tensors"
                                   : "; seqlens_k not found, falling back to full KV output"));
        for (size_t ki = 0; ki < kv_indices.size(); ++ki) {
            LOG_INFO("  KV[" << ki << "] output_idx=" << kv_indices[ki] << " max_seq=" << kv_max_seqs[ki]
                             << (kv_transposed_flags[ki] ? " (transposed V)" : ""));
        }

        // When KV management is not possible, fall back to using the full-output (outer) model
        // for inner compilation so set_tensor forwarding stays shape-consistent.
        auto inner = can_manage_kv ? model : nullptr;
        return {std::move(outer_model),
                std::move(inner),
                std::move(prepared_properties),
                can_manage_kv,
                std::move(seqlens_k_name),
                std::move(kv_indices),
                std::move(kv_max_seqs),
                std::move(kv_transposed_flags)};
    }
    return {model, nullptr, std::move(prepared_properties), false, {}, {}, {}};
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::GQACompiledModel::make_compiled_model(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    return std::make_shared<ov::npuw::CompiledModel>(model, plugin, properties);
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
      m_compiled_model(
          factory(prepared.inner_model ? prepared.inner_model : prepared.model, plugin, prepared.properties)),
      m_kv_managed(prepared.kv_managed),
      m_seqlens_k_name(std::move(prepared.seqlens_k_name)),
      m_kv_output_indices(std::move(prepared.kv_output_indices)),
      m_kv_max_seqs(std::move(prepared.kv_max_seqs)),
      m_kv_transposed(std::move(prepared.kv_transposed)) {
    OPENVINO_ASSERT(m_compiled_model != nullptr, "GQACompiledModel requires a valid inner compiled model");
    // When the stub outer model has extra (stub) Parameters appended after the real inputs,
    // build m_outer_inputs so that inputs() hides them from the user-facing interface.
    if (prepared.real_input_count > 0) {
        const auto& all_inputs = ICompiledModel::inputs();
        OPENVINO_ASSERT(prepared.real_input_count <= all_inputs.size(),
                        "real_input_count exceeds total inputs in stub outer model");
        m_outer_inputs.assign(all_inputs.begin(), all_inputs.begin() + prepared.real_input_count);
    }
}

void ov::npuw::GQACompiledModel::export_model(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write(stream, NPUW_SERIALIZATION_INDICATOR);
    write(stream, NPUW_GQA_COMPILED_MODEL_INDICATOR);
    write(stream, OPENVINO_VERSION_MAJOR);
    write(stream, OPENVINO_VERSION_MINOR);
    write(stream, OPENVINO_VERSION_PATCH);
    write(stream, std::string(NPUW_SERIALIZATION_VERSION));
    // KV scatter management metadata — must be read back in import_model before the inner blob.
    write(stream, m_kv_managed);
    if (m_kv_managed) {
        write(stream, m_seqlens_k_name);
        write(stream, m_kv_output_indices.size());
        for (size_t i = 0; i < m_kv_output_indices.size(); ++i) {
            write(stream, m_kv_output_indices[i]);
            write(stream, m_kv_max_seqs[i]);
            bool transposed = (i < m_kv_transposed.size()) && m_kv_transposed[i];
            write(stream, transposed);
        }
    }
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

    // Read KV scatter management metadata (written by export_model before the inner blob).
    bool kv_managed = false;
    read(stream, kv_managed);

    std::string seqlens_k_name;
    std::vector<size_t> kv_indices;
    std::vector<size_t> kv_max_seqs;
    std::vector<bool> kv_transposed;
    if (kv_managed) {
        read(stream, seqlens_k_name);
        size_t count = 0;
        read(stream, count);
        kv_indices.reserve(count);
        kv_max_seqs.reserve(count);
        kv_transposed.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            size_t idx = 0, max_seq = 0;
            bool trans = false;
            read(stream, idx);
            read(stream, max_seq);
            read(stream, trans);
            kv_indices.push_back(idx);
            kv_max_seqs.push_back(max_seq);
            kv_transposed.push_back(trans);
        }
    }

    // Load the inner compiled model blob.
    auto inner_cm = ov::npuw::CompiledModel::import_model(stream, plugin, properties);

    if (!kv_managed) {
        // No KV management: the inner model IS the full model, return as-is.
        return inner_cm;
    }

    LOG_INFO("Reconstructing GQACompiledModel wrapper with KV scatter management ("
             << kv_indices.size() << " KV outputs, seqlens_k='" << seqlens_k_name << "')");

    // Build a stub outer model that exposes the user-facing shapes (KV outputs with max_seq).
    auto outer_model = build_stub_outer_model(inner_cm, kv_indices, kv_max_seqs, kv_transposed);

    // The factory ignores its model argument and returns the already-loaded inner compiled model.
    CompiledModelFactory factory = [inner_cm](const std::shared_ptr<ov::Model>&,
                                              const std::shared_ptr<const ov::IPlugin>&,
                                              const ov::AnyMap&) -> std::shared_ptr<ov::npuw::ICompiledModel> {
        return inner_cm;
    };

    PreparedState ps;
    ps.model = outer_model;
    ps.inner_model = nullptr;  // unused: factory ignores model argument
    ps.properties = properties;
    ps.kv_managed = true;
    ps.seqlens_k_name = std::move(seqlens_k_name);
    ps.kv_output_indices = std::move(kv_indices);
    ps.kv_max_seqs = std::move(kv_max_seqs);
    ps.kv_transposed = std::move(kv_transposed);
    // Tell the constructor how many leading Parameters in the stub model are REAL inputs.
    // The remainder are stub Parameters for outputs (appended by build_stub_outer_model).
    ps.real_input_count = inner_cm->inputs().size();
    return std::shared_ptr<GQACompiledModel>(new GQACompiledModel(std::move(ps), plugin, std::move(factory)));
}

std::shared_ptr<const ov::Model> ov::npuw::GQACompiledModel::get_runtime_model() const {
    return m_compiled_model->get_runtime_model();
}

const std::vector<ov::Output<const ov::Node>>& ov::npuw::GQACompiledModel::inputs() const {
    // When the stub outer model was built with extra stub Parameters (import_model path),
    // m_outer_inputs holds only the REAL model inputs.  Return it instead of the base-class
    // m_inputs which would include the stub Parameters as spurious extra inputs.
    if (!m_outer_inputs.empty())
        return m_outer_inputs;
    return ICompiledModel::inputs();
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

bool ov::npuw::GQAInferRequest::is_kv_output_locked(size_t idx) const {
    if (!m_compiled_model->m_kv_managed)
        return false;
    const auto& kv = m_compiled_model->m_kv_output_indices;
    return std::find(kv.begin(), kv.end(), idx) != kv.end();
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

    // When the KV scatter is managed here, read seqlens_k before the inner infer so we
    // can compute the write offset into the user-supplied full KV-cache tensors.
    int64_t seqlens_k_val = 0;
    if (m_compiled_model->m_kv_managed) {
        const auto& inner_inputs = m_inner_request->get_compiled_model()->inputs();
        size_t sk_idx = SIZE_MAX;
        for (size_t i = 0; i < inner_inputs.size(); ++i) {
            if (inner_inputs[i].get_node()->get_friendly_name() == m_compiled_model->m_seqlens_k_name) {
                sk_idx = i;
                break;
            }
        }
        OPENVINO_ASSERT(sk_idx != SIZE_MAX,
                        "seqlens_k port '",
                        m_compiled_model->m_seqlens_k_name,
                        "' not found in inner compiled model");
        auto sk = m_inner_request->get_tensor(inner_inputs[sk_idx]);
        if (sk->get_element_type() == ov::element::i32)
            seqlens_k_val = static_cast<int64_t>(*reinterpret_cast<const int32_t*>(sk->data()));
        else
            seqlens_k_val = *reinterpret_cast<const int64_t*>(sk->data());
    }

    m_inner_request->infer();

    if (!m_compiled_model->m_kv_managed)
        return;

    // Scatter the small inner KV outputs into the right offset of the user's full KV tensors.
    for (size_t ki = 0; ki < m_compiled_model->m_kv_output_indices.size(); ++ki) {
        const size_t kv_idx = m_compiled_model->m_kv_output_indices[ki];
        auto user_it = m_user_kv_tensors.find(kv_idx);
        auto work_it = m_kv_working_tensors.find(kv_idx);
        if (user_it == m_user_kv_tensors.end() || work_it == m_kv_working_tensors.end())
            continue;

        // inner_t is the working tensor NPUW wrote K curr data into.
        auto& inner_t = work_it->second;
        auto& user_t = user_it->second;

        // inner_t shape (all KV): [1, kv_heads, curr_seq, head_size]
        const auto inner_shape = inner_t->get_shape();
        const size_t kv_heads = inner_shape[1];
        const size_t curr_seq = inner_shape[2];
        const size_t head_size = inner_shape[3];
        const size_t past = static_cast<size_t>(seqlens_k_val);
        const size_t esz = inner_t->get_element_type().size();

        const char* src = reinterpret_cast<const char*>(inner_t->data());
        char* dst = reinterpret_cast<char*>(user_t->data());

        const bool transposed =
            (ki < m_compiled_model->m_kv_transposed.size()) && m_compiled_model->m_kv_transposed[ki];
        if (!transposed) {
            // Standard K layout: user_t is [1, kv_heads, max_seq, head_size].
            // Copy curr_seq contiguous tokens per head at sequence offset `past`.
            const size_t max_seq = user_t->get_shape()[2];
            for (size_t h = 0; h < kv_heads; ++h) {
                std::memcpy(dst + (h * max_seq + past) * head_size * esz,
                            src + h * curr_seq * head_size * esz,
                            curr_seq * head_size * esz);
            }
        } else {
            // Transposed V layout: user_t is [1, kv_heads, head_size, max_seq].
            // inner_t[h, t, d] → user_t[h, d, past + t]
            // The target stride along max_seq is contiguous, so we scatter
            // one element at a time (or one token-column per head_dim).
            const size_t max_seq = user_t->get_shape()[3];
            for (size_t h = 0; h < kv_heads; ++h) {
                for (size_t d = 0; d < head_size; ++d) {
                    // Destination: user_t column [h, d, past .. past+curr_seq-1] — contiguous.
                    char* dst_col = dst + ((h * head_size + d) * max_seq + past) * esz;
                    for (size_t t = 0; t < curr_seq; ++t) {
                        std::memcpy(dst_col + t * esz, src + ((h * curr_seq + t) * head_size + d) * esz, esz);
                    }
                }
            }
        }
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::GQAInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    if (m_compiled_model->m_kv_managed) {
        const auto& outer_outputs = m_compiled_model->outputs();
        for (size_t i = 0; i < outer_outputs.size(); ++i) {
            if (outer_outputs[i] != port || !is_kv_output_locked(i))
                continue;
            auto it = m_user_kv_tensors.find(i);
            if (it != m_user_kv_tensors.end())
                return it->second;
            // Allocate a default tensor from the outer port's static shape.
            auto t =
                ov::make_tensor(outer_outputs[i].get_element_type(), outer_outputs[i].get_partial_shape().to_shape());
            m_user_kv_tensors[i] = t;
            return t;
        }
    }
    return m_inner_request->get_tensor(map_port_locked(port));
}

void ov::npuw::GQAInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                           const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();

    if (m_compiled_model->m_kv_managed) {
        const auto& outer_outputs = m_compiled_model->outputs();
        const auto& inner_outputs = m_inner_request->get_compiled_model()->outputs();
        for (size_t i = 0; i < outer_outputs.size(); ++i) {
            if (outer_outputs[i] == port && is_kv_output_locked(i)) {
                // Store the user's full KV tensor (scatter destination).
                m_user_kv_tensors[i] = tensor;
                // Provide the inner request with a correctly-sized working buffer so
                // NPUW never has to allocate its own — the tensor is always set from outside.
                if (i < inner_outputs.size()) {
                    auto& wt = m_kv_working_tensors[i];
                    if (!wt) {
                        const auto& inner_port = inner_outputs[i];
                        wt = ov::make_tensor(inner_port.get_element_type(), inner_port.get_partial_shape().to_shape());
                    }
                    m_inner_request->set_tensor(inner_outputs[i], wt);
                }
                return;
            }
        }
    }
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
