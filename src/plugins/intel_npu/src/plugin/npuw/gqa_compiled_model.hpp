// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "npuw/compiled_model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::npuw {

class GQACompiledModel;

class GQAInferRequest final : public ov::ISyncInferRequest {
public:
    explicit GQAInferRequest(std::shared_ptr<const GQACompiledModel> compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // Best-effort match of a present_key/present_value output's friendly name back to
    // its corresponding past_key/past_value input's friendly name, by swapping the
    // "present" substring (case-insensitive) for "past". `name` may still carry a
    // node-name suffix (e.g. the ONNX frontend's "/sink_port_0"); it is stripped
    // before matching. Returns nullopt if "present" isn't found. Exposed for testing.
    static std::optional<std::string> present_to_past_name(const std::string& name);

    // Copies the valid prefix of a smaller, dynamically-sized rank-4 tensor (KV-cache or
    // attention bias) into a larger statically-shaped one, left-aligned, along `axis` (2
    // for [N,H,S,E], 3 for the transpose_v-applied [N,H,E,S]/[N,H,X,S] layout). Exposed
    // for testing.
    static void copy_kv_cache_prefix(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst, size_t axis);

private:
    void ensure_inner_request_locked() const;
    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;

    std::shared_ptr<const GQACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_inner_request;
    // Outer-facing tensor for dynamic KV-cache/attention-bias ports (see
    // GQACompiledModel::m_dynamic_kv_cache_axes); only its valid prefix is copied into
    // the inner request's static buffer. Keyed by port friendly name.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_tensors;
    // Names of dynamic-axis inputs currently bound directly onto the caller's own
    // (exact-capacity) tensor inside the inner request, rather than a private buffer.
    // set_tensor() with a shorter tensor on the same port must restore the stashed
    // private buffer below before the inner request is used again.
    mutable std::unordered_set<std::string> m_dynamic_kv_cache_input_aliased;
    // The inner request's own, privately-owned static-capacity buffer for a dynamic-axis
    // input, stashed away the first time that port gets aliased directly onto a caller's
    // exact-capacity tensor (see set_tensor()); restored once a shorter tensor is bound
    // to the same port, so a stale caller tensor from a previous call is never written
    // into or left dangling inside the inner request.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_private_buffers;
    // Outer-facing tensors for dynamic present_key/present_value outputs; allocated
    // lazily and refreshed from the inner static buffer after every infer().
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_output_tensors;
    // Names currently aliased directly onto the inner request's own tensor (no
    // trimming needed) rather than a private buffer; never set_shape() these.
    mutable std::unordered_set<std::string> m_dynamic_kv_cache_output_aliased;
    // Present-output names the caller explicitly set_tensor()'d (IO-binding style):
    // forwarded straight to the inner request's own port, so refresh_present_tensors_locked()
    // must not republish a separately-trimmed copy under this name afterwards.
    mutable std::unordered_set<std::string> m_dynamic_kv_cache_output_externally_set;

    ov::SoPtr<ov::ITensor> get_present_tensor_locked(const std::string& name) const;
    // Reads the live value of the Parameter feeding the GQA op's TOTAL_SEQUENCE_LENGTH
    // input, if one was resolved at compile time; see gqa_compiled_model.cpp.
    std::optional<size_t> read_total_sequence_length_locked() const;
    void refresh_present_tensors_locked() const;
    // Copies each dynamic-axis input's user-owned tensor into the inner request's
    // static buffer, right before infer() (not inside set_tensor(), since the caller
    // may keep writing into it afterwards). No-op for ports already aliased.
    void sync_dynamic_kv_cache_tensors_locked() const;
    // Diagnostic-only: traces sequence-length-style inputs (seqlens_k / past_seq_len /
    // total_seq_len) right before infer(); see gqa_compiled_model.cpp.
    void trace_sequence_length_inputs_locked() const;
    // Diagnostic-only: traces attention bias/mask stats right before infer().
    void trace_attention_mask_stats_locked() const;
};

class GQACompiledModel final : public ov::npuw::ICompiledModel {
public:
    using CompiledModelFactory =
        std::function<std::shared_ptr<ov::npuw::ICompiledModel>(const std::shared_ptr<ov::Model>&,
                                                                const std::shared_ptr<const ov::IPlugin>&,
                                                                const ov::AnyMap&)>;

    enum class Case {
        Unknown,
        V0,
        V1,
    };

    static std::shared_ptr<ov::npuw::ICompiledModel> make_compiled_model(
        const std::shared_ptr<ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin,
        const ov::AnyMap& properties);

    // Identifies which known GQA model family (if any) `model` belongs to, purely from its
    // Parameter/Result/op structure -- independent of whether any dimension is dynamic.
    static Case identify_case(const std::shared_ptr<const ov::Model>& model);

    // True if `model` should be auto-dispatched to GQACompiledModel: it must match a known
    // GQA family (identify_case() != Case::Unknown) AND actually have a dynamic max_seq_len
    // (has_dynamic_max_seq_len()) -- a fully static model of a known family doesn't need this
    // wrapper.
    static bool supports(const std::shared_ptr<const ov::Model>& model);

    // True if any GroupQueryAttention op's past_key/past_value or attention bias input has a
    // dynamic (unbounded) max_seq_len dimension, which the NPU compiler cannot handle directly.
    static bool has_dynamic_max_seq_len(const std::shared_ptr<const ov::Model>& model);

    // Scans `outer_outputs` for present_key/present_value ports with a single dynamic
    // axis (the output-side mirror of the past_key/past_value scan done for
    // m_dynamic_kv_cache_axes). Re-derivable from outer output ports alone (no
    // serialization needed), so it's recomputed on both the compile and import paths.
    // Exposed for testing.
    static std::unordered_map<std::string, size_t> find_dynamic_kv_cache_output_axes(
        const std::vector<ov::Output<const ov::Node>>& outer_outputs);

    // Writes/reads outer-facing port metadata (friendly name + element type + partial
    // shape) for export_model()/import_model(). Deliberately keyed by friendly name (not
    // tensor names, which the generic NPUW serialize() overloads for ov::Output<const
    // ov::Node>/Parameter/Node rely on and which aren't guaranteed to be set) because
    // GQAInferRequest's dynamic-axis lookups key off get_friendly_name() too. Exposed
    // publicly (static) so the wire-format round trip can be exercised directly in tests.
    static void write_port_list(std::ostream& stream, const std::vector<ov::Output<const ov::Node>>& ports);
    static ov::ParameterVector read_input_port_list(std::istream& stream);
    static ov::NodeVector read_output_port_list(std::istream& stream);

    GQACompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const ov::AnyMap& properties,
                     CompiledModelFactory factory = make_compiled_model);

    static std::shared_ptr<ov::npuw::ICompiledModel> import_model(std::istream& stream,
                                                                  const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                  const ov::AnyMap& properties);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    struct PreparedState {
        std::shared_ptr<ov::Model> model;
        std::shared_ptr<ov::Model> compiled_model;
        ov::AnyMap properties;
        // KV-cache/attention-bias Parameter friendly name -> axis pinned to kMaxSeqLen
        // (only populated when has_dynamic_max_seq_len() is true for the model).
        std::unordered_map<std::string, size_t> dynamic_kv_cache_axes;
        // The static KV-cache capacity actually used to reshape the model (either the
        // OPENVINO_NPUW_GQA_CTX_LEN env override or the NPUW_LLM_MAX_CONTEXT_LEN
        // property; see prepare()). NPUW_LLM_MAX_CONTEXT_LEN itself is stripped from
        // the inner CompiledModel's properties (it only registers ROOT options), so
        // this is the only way to read the effective capacity back afterwards.
        uint32_t effective_max_context_len = 0;
    };

    static PreparedState prepare(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& properties);

    GQACompiledModel(PreparedState prepared,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     CompiledModelFactory factory);

    // Used by import_model() to reconstruct a GQACompiledModel from an
    // already-deserialized inner compiled model (see gqa_compiled_model.cpp).
    GQACompiledModel(const std::shared_ptr<ov::Model>& outer_model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     std::shared_ptr<ov::npuw::ICompiledModel> inner_compiled_model,
                     std::unordered_map<std::string, size_t> dynamic_kv_cache_axes,
                     uint32_t effective_max_context_len);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    friend class GQAInferRequest;

    std::shared_ptr<ov::npuw::ICompiledModel> m_inner_compiled_model;
    // The outer-facing model (Parameters/Results as seen by the caller, dynamic
    // KV-cache/attention-bias axes as originally presented -- i.e. before the
    // reshape-to-static done in prepare()). Set on both the compile path (the
    // pre-reshape PreparedState::model) and the import path (rebuilt from the
    // exported port lists). Returned as-is from get_runtime_model(): callers care
    // about the outer contract, not the inner (already-partitioned) implementation.
    std::shared_ptr<const ov::Model> m_outer_model;
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_axes;
    // present_key/present_value outer output friendly name -> axis. Unlike
    // m_dynamic_kv_cache_axes this is *not* serialized: it's cheaply re-derivable from
    // the (already round-tripped) outer output ports on both the compile and import
    // paths, so there's no need to grow the wire format for it.
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_output_axes;
    // Friendly name of the Parameter feeding the GroupQueryAttention op's
    // TOTAL_SEQUENCE_LENGTH input, if one can be traced back from it (nullopt
    // otherwise). Like m_dynamic_kv_cache_output_axes, this is re-derived from
    // m_outer_model on both the compile and import paths rather than serialized.
    // Used to compute an accurate present_key/present_value length (past length
    // "grown" by the newly appended tokens) instead of just mirroring past_key/
    // past_value's own length, which misses the rows appended by the current step.
    std::optional<std::string> m_total_seq_len_param_name;
    // The effective static KV-cache capacity (see PreparedState::effective_max_context_len
    // above); returned from get_property(NPUW_LLM_MAX_CONTEXT_LEN) since the inner
    // CompiledModel no longer carries this property (see prepare()). Serialized across
    // export_model()/import_model() so a cache hit reports the same value.
    uint32_t m_effective_max_context_len = 0;
};

}  // namespace ov::npuw
