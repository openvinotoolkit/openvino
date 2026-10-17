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

    // Copies the valid prefix of a smaller rank-4 tensor into a larger static one,
    // left-aligned along `axis` (2 for [N,H,S,E], 3 for transposed [N,H,E,S]).
    static void copy_kv_cache_prefix(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst, size_t axis);

private:
    void ensure_inner_request_locked() const;
    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;

    std::shared_ptr<const GQACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_inner_request;
    // Outer-facing tensor for dynamic KV-cache/attention-bias ports, keyed by port name.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_tensors;
    // Dynamic-axis inputs currently bound directly onto the caller's exact-capacity
    // tensor; must be restored before a shorter set_tensor() reuses the port.
    mutable std::unordered_set<std::string> m_dynamic_kv_cache_input_aliased;
    // Private static-capacity buffer stashed when a port gets aliased (see above);
    // restored once a shorter tensor is bound to the same port.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_private_buffers;
    // Outer-facing tensors for dynamic present_key/present_value outputs; allocated
    // lazily and refreshed from the inner static buffer after every infer().
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_output_tensors;
    // Names currently aliased directly onto the inner tensor (no copy needed because the
    // whole buffer is valid); never set_shape() these.
    mutable std::unordered_set<std::string> m_dynamic_kv_cache_output_aliased;
    // Present-output names the caller explicitly set_tensor()'d (forwarded straight to
    // the inner request after asserting an exact shape match -- see set_tensor());
    // refresh_present_tensors_locked() must not republish a separately-copied valid
    // prefix under these names.
    mutable std::unordered_set<std::string> m_dynamic_kv_cache_output_externally_set;

    ov::SoPtr<ov::ITensor> get_present_tensor_locked(const std::string& name) const;
    // Reads the live TOTAL_SEQUENCE_LENGTH Parameter value; see gqa_compiled_model.cpp.
    std::optional<size_t> read_total_sequence_length_locked() const;
    void refresh_present_tensors_locked() const;
    // Copies each dynamic-axis input into the inner request's static buffer right before
    // infer(). No-op for ports already aliased.
    void sync_dynamic_kv_cache_tensors_locked() const;
    // Diagnostic-only: traces sequence-length-style inputs before infer().
    void trace_sequence_length_inputs_locked() const;
    // Diagnostic-only: traces attention bias/mask stats before infer().
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
        V2,
    };

    static std::shared_ptr<ov::npuw::ICompiledModel> make_compiled_model(
        const std::shared_ptr<ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin,
        const ov::AnyMap& properties);

    // Classifies `model`'s GQA family, independent of dynamic shapes.
    static Case identify_case(const std::shared_ptr<const ov::Model>& model);

    // True if `model` should auto-dispatch to GQACompiledModel: a known GQA family with
    // an actually dynamic max_seq_len.
    static bool supports(const std::shared_ptr<const ov::Model>& model);

    // True if any GQA op's KV-cache/bias input has a dynamic max_seq_len dimension.
    static bool has_dynamic_max_seq_len(const std::shared_ptr<const ov::Model>& model);

    // Scans `outer_outputs` for present_key/present_value ports with a single dynamic
    // axis (output-side mirror of m_dynamic_kv_cache_axes). Exposed for testing.
    static std::unordered_map<std::string, size_t> find_dynamic_kv_cache_output_axes(
        const std::vector<ov::Output<const ov::Node>>& outer_outputs);

    // Maps each present_key/present_value Result's name to its past_key/past_value
    // Parameter's name, derived from each GQA op's own graph connections (not string
    // rewriting, which breaks for dot-style naming). Exposed for testing.
    static std::unordered_map<std::string, std::string> find_present_to_past_param_names(
        const std::shared_ptr<const ov::Model>& model);

    // Writes/reads outer-facing port metadata for export_model()/import_model(), keyed by
    // friendly name (GQAInferRequest's dynamic-axis lookups key off it too).
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
        // KV-cache/attention-bias Parameter name -> axis pinned to kMaxSeqLen.
        std::unordered_map<std::string, size_t> dynamic_kv_cache_axes;
        // Static KV-cache capacity actually used to reshape the model; stripped from the
        // inner CompiledModel's properties, so this is the only way to read it back.
        uint32_t effective_max_context_len = 0;
        // Resolved sliding-window fallback capacity (see resolve_default_sliding_window_capacity()),
        // 0 if NPUW_LLM_SWA_LEN wasn't configured. Also stripped from the inner
        // CompiledModel's properties.
        uint32_t effective_sliding_window_capacity = 0;
    };

    static PreparedState prepare(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& properties);

    GQACompiledModel(PreparedState prepared,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     CompiledModelFactory factory);

    // Used by import_model() to reconstruct a GQACompiledModel from a deserialized inner
    // compiled model. total_seq_len_param_name/present_to_past_param_names are passed in
    // (not re-derived from outer_model): on import, outer_model is a dummy carcass with
    // no actual GroupQueryAttention ops, so the graph-walking find_*() helpers would
    // silently find nothing.
    GQACompiledModel(const std::shared_ptr<ov::Model>& outer_model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     std::shared_ptr<ov::npuw::ICompiledModel> inner_compiled_model,
                     std::unordered_map<std::string, size_t> dynamic_kv_cache_axes,
                     uint32_t effective_max_context_len,
                     uint32_t effective_sliding_window_capacity,
                     std::optional<std::string> total_seq_len_param_name,
                     std::unordered_map<std::string, std::string> present_to_past_param_names);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    friend class GQAInferRequest;

    std::shared_ptr<ov::npuw::ICompiledModel> m_inner_compiled_model;
    // The outer-facing model (as seen by the caller, pre-reshape-to-static). Returned
    // as-is from get_runtime_model(); callers care about the outer contract.
    std::shared_ptr<const ov::Model> m_outer_model;
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_axes;
    // present_key/present_value outer output name -> axis. Not serialized: cheaply
    // re-derivable from the outer output ports on both compile and import paths.
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_output_axes;
    // TOTAL_SEQUENCE_LENGTH Parameter name, if resolved. Used to compute an accurate
    // present_key/present_value length instead of mirroring past_key/past_value's own
    // (stale) length. Derived from the real model on first compile and serialized across
    // export_model()/import_model() -- NOT re-derivable from m_outer_model on import,
    // since the import path's outer_model is a dummy carcass with no real GQA ops.
    std::optional<std::string> m_total_seq_len_param_name;
    // present_key/present_value outer Result name -> its past_key/past_value Parameter
    // name (see find_present_to_past_param_names()). Same as above: derived once from the
    // real model, serialized, and passed through on import rather than re-derived.
    std::unordered_map<std::string, std::string> m_present_to_past_param_names;
    // Effective static KV-cache capacity; returned from get_property(NPUW_LLM_MAX_CONTEXT_LEN)
    // since the inner CompiledModel no longer carries this property. Serialized across
    // export_model()/import_model().
    uint32_t m_effective_max_context_len = 0;
    // Effective sliding-window fallback capacity (0 if unconfigured); returned from
    // get_property(NPUW_LLM_SWA_LEN) for the same reason. Serialized across
    // export_model()/import_model().
    uint32_t m_effective_sliding_window_capacity = 0;
};

}  // namespace ov::npuw
