// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "npuw/compiled_model.hpp"

namespace ov::npuw {

class GQACompiledModel;

class GQAInferRequest : public ov::ISyncInferRequest {
public:
    explicit GQAInferRequest(std::shared_ptr<const GQACompiledModel> compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

protected:
    void ensure_inner_request_locked() const;
    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;

    std::shared_ptr<const GQACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_inner_request;
};

class ManagedGQAInferRequest final : public GQAInferRequest {
public:
    explicit ManagedGQAInferRequest(std::shared_ptr<const GQACompiledModel> compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

private:
    bool is_kv_output_locked(size_t output_index) const;
    int64_t read_seqlens_k_locked() const;
    void scatter_kv_outputs_locked(int64_t seqlens_k_val) const;

    // User-set tensors for KV outputs intercepted by the managed KV scatter path.
    mutable std::unordered_map<size_t, ov::SoPtr<ov::ITensor>> m_user_kv_tensors;
    // Correctly-sized working tensors forwarded to the inner request for each managed KV output.
    mutable std::unordered_map<size_t, ov::SoPtr<ov::ITensor>> m_kv_working_tensors;
};

class GQACompiledModel final : public ov::npuw::ICompiledModel {
public:
    using CompiledModelFactory =
        std::function<std::shared_ptr<ov::npuw::ICompiledModel>(const std::shared_ptr<ov::Model>&,
                                                                const std::shared_ptr<const ov::IPlugin>&,
                                                                const ov::AnyMap&)>;

    static std::shared_ptr<ov::npuw::ICompiledModel> make_compiled_model(
        const std::shared_ptr<ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin,
        const ov::AnyMap& properties);

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
        std::shared_ptr<ov::Model> model;        // outer interface (full KV output shapes)
        std::shared_ptr<ov::Model> inner_model;  // for NPU compilation (stripped KV outputs)
        ov::AnyMap properties;
        bool sliced = false;                        // true when the wrapper manages KV slicing/scatter itself
        std::string seqlens_k_name;                 // friendly name of the seqlens_k parameter
        std::vector<size_t> sliced_output_indices;  // which result indices are sliced by the wrapper
        std::vector<size_t> sliced_max_seqs;        // max_seq for each sliced output
        std::vector<bool> sliced_transposed;        // true when V output is transposed [1,H,head_size,max_seq]
        // When importing from blob the stub model includes extra Parameters for outputs
        // (to satisfy ov::Model validation).  This field records how many leading Parameters
        // are REAL model inputs; inputs() overrides the base-class list accordingly.
        size_t real_input_count = 0;
    };

    static PreparedState prepare(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& properties);

    GQACompiledModel(PreparedState prepared,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     CompiledModelFactory factory);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    // Override inputs() to hide the extra stub Parameters that the stub outer model
    // (built during import_model) includes to satisfy ov::Model validation.
    // In the normal compile path m_outer_inputs is empty, so the base class result is used.
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;

    friend class GQAInferRequest;
    friend class ManagedGQAInferRequest;

    std::shared_ptr<ov::npuw::ICompiledModel> m_compiled_model;
    bool m_sliced = false;
    std::string m_seqlens_k_name;
    std::vector<size_t> m_sliced_output_indices;
    std::vector<size_t> m_sliced_max_seqs;
    std::vector<bool> m_sliced_transposed;  // per-output: true when V is [1,H,head_size,max_seq]
    // Populated during import_model when the stub outer model has extra stub Parameters.
    // inputs() returns this instead of the base-class m_inputs when non-empty.
    std::vector<ov::Output<const ov::Node>> m_outer_inputs;
};

}  // namespace ov::npuw
