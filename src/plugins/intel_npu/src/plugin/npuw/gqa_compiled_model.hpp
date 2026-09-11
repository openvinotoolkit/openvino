// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "npuw/compiled_model.hpp"

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
    // For dynamic KV-cache ports (see GQACompiledModel::m_dynamic_kv_cache_axis), the
    // outer-facing tensor is user-owned and only its valid prefix is copied into the
    // inner request's static buffer; get_tensor() must hand back this exact tensor
    // rather than the inner (differently-shaped) one. Keyed by port friendly name.
    mutable std::unordered_map<std::string, ov::SoPtr<ov::ITensor>> m_dynamic_kv_cache_tensors;
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

    // Identifies which known GQA model family (if any) `model` belongs to.
    static Case identify_case(const std::shared_ptr<const ov::Model>& model);

    // True if `model` matches any known GQA family, i.e. identify_case() != Case::Unknown.
    static bool supports(const std::shared_ptr<const ov::Model>& model);

    // True if any GroupQueryAttention op's past_key/past_value input has a dynamic
    // (unbounded) max_seq_len dimension, which the NPU compiler cannot handle directly.
    static bool has_dynamic_max_seq_len(const std::shared_ptr<const ov::Model>& model);

    // Copies the valid prefix of a smaller, dynamically-sized KV-cache tensor into a
    // larger statically-shaped one, left-aligned, along `axis` (2 for [N,H,S,E], 3 for
    // the transpose_v-applied [N,H,E,S] layout). Both must be rank-4 KV-cache tensors.
    static void copy_kv_cache_prefix(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst, size_t axis);

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
        // KV-cache Parameter friendly name -> axis pinned to kMaxSeqLen (only
        // populated when has_dynamic_max_seq_len() is true for the model).
        std::unordered_map<std::string, size_t> dynamic_kv_cache_axis;
    };

    static PreparedState prepare(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& properties);

    GQACompiledModel(PreparedState prepared,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     CompiledModelFactory factory);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    friend class GQAInferRequest;

    std::shared_ptr<ov::npuw::ICompiledModel> m_compiled_model;
    std::unordered_map<std::string, size_t> m_dynamic_kv_cache_axis;
};

}  // namespace ov::npuw
