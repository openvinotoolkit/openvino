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

private:
    void ensure_inner_request_locked() const;
    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;

    std::shared_ptr<const GQACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_inner_request;
    // For dynamic KV-cache/attention-bias ports (see GQACompiledModel::m_dynamic_kv_cache_axes),
    // the outer-facing tensor is user-owned and only its valid prefix is copied into the
    // inner request's static buffer; get_tensor() must hand back this exact tensor
    // rather than the inner (differently-shaped) one. Keyed by port friendly name.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_tensors;
    // Outer-facing tensors for dynamic present_key/present_value outputs (see
    // GQACompiledModel::m_dynamic_kv_cache_output_axes): allocated lazily on first
    // get_tensor() and refreshed (valid prefix copied out of the inner static buffer)
    // at the end of every infer(). Keyed by port friendly name.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_output_tensors;

    ov::SoPtr<ov::ITensor> get_present_tensor_locked(const std::string& name, size_t axis) const;
    void refresh_present_tensors_locked() const;
    // Copies the live contents of each dynamic-axis KV-cache/attention-bias input's
    // user-owned tensor (captured by set_tensor(), see m_dynamic_kv_cache_tensors) into
    // the inner request's static buffer. Must run right before infer() delegates to the
    // inner request, not inside set_tensor() itself: a caller may keep writing into the
    // same tensor object after set_tensor() and before infer() (e.g. set_tensor(t);
    // write_into(t); infer();), and only data present at infer() time is guaranteed to
    // be picked up.
    void sync_dynamic_kv_cache_tensors_locked() const;
    // Diagnostic-only: prints the current values of any sequence-length-style input
    // this model exposes (seqlens_k / past_seq_len / total_seq_len), read fresh from
    // the inner request right before infer() -- see gqa_compiled_model.cpp for why this
    // needs to look at the tensor's live content rather than trust set_tensor() alone.
    void trace_sequence_length_inputs_locked() const;
    // Diagnostic-only: prints min/max/non-zero-count for the attention bias/mask
    // input(s), read fresh from the inner request right before infer() -- answers
    // whether this model actually relies on external mask content (vs. leaving it
    // permanently zero/inert and relying purely on GQA's internal causal masking).
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

    // Best-effort match of a present_key/present_value output's friendly name back to
    // its corresponding past_key/past_value input's friendly name, by swapping the
    // "present" substring (case-insensitive) for "past". `name` may still carry a
    // node-name suffix (e.g. the ONNX frontend's "/sink_port_0"); it is stripped
    // before matching. Returns nullopt if "present" isn't found. Exposed for testing.
    static std::optional<std::string> present_to_past_name(const std::string& name);

    // Copies the valid prefix of a smaller, dynamically-sized rank-4 tensor (KV-cache or
    // attention bias) into a larger statically-shaped one, left-aligned, along `axis` (2
    // for [N,H,S,E], 3 for the transpose_v-applied [N,H,E,S]/[N,H,X,S] layout).
    static void copy_kv_cache_prefix(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst, size_t axis);

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
                     std::unordered_map<std::string, size_t> dynamic_kv_cache_axes);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    friend class GQAInferRequest;

    std::shared_ptr<ov::npuw::ICompiledModel> m_inner_compiled_model;
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_axes;
    // present_key/present_value outer output friendly name -> axis. Unlike
    // m_dynamic_kv_cache_axes this is *not* serialized: it's cheaply re-derivable from
    // the (already round-tripped) outer output ports on both the compile and import
    // paths, so there's no need to grow the wire format for it.
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_output_axes;
};

}  // namespace ov::npuw
