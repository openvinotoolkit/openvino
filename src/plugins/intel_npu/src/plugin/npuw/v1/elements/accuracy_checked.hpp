// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov::npuw::accuracy_checked {

class InferRequest;

// A compiled-model wrapper that validates inference accuracy against a
// reference device and permanently switches to that reference if the
// normalised RMSE threshold is exceeded.
//
// The wrapper is transparent: it exposes the same I/O as the wrapped
// model and forwards all property queries to the currently active model
// (main until a switch happens, reference afterwards).
//
// Intended to be composed on top of a failsafe::CompiledModel so that
// the full chain becomes:
//
//   AccuracyChecked( main = Failsafe(NPU -> CPU), ref = CPU )
class CompiledModel final : public ov::ICompiledModel {
public:
    // Checker: returns true when the output pair is considered accurate.
    using Checker = std::function<bool(const ov::SoPtr<ov::ITensor>&, const ov::SoPtr<ov::ITensor>&)>;

    // Factory method.  Returns main_compiled unwrapped when ref_compiled is
    // null (no-op wrapper) to keep the zero-overhead path trivial.
    static ov::SoPtr<ov::ICompiledModel> create(const std::shared_ptr<ov::Model>& model,
                                                const std::shared_ptr<const ov::IPlugin>& plugin,
                                                ov::SoPtr<ov::ICompiledModel> main_compiled,
                                                ov::SoPtr<ov::ICompiledModel> ref_compiled,
                                                Checker checker);

    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  ov::SoPtr<ov::ICompiledModel> main_compiled,
                  ov::SoPtr<ov::ICompiledModel> ref_compiled,
                  Checker checker);

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    // Returns true if any InferRequest has triggered a permanent switch
    // to the reference compiled model.
    bool has_switched_to_reference() const;

private:
    friend class InferRequest;

    ov::SoPtr<ov::ICompiledModel> active_compiled_model_locked() const;

    ov::SoPtr<ov::ICompiledModel> m_main_compiled;
    ov::SoPtr<ov::ICompiledModel> m_ref_compiled;
    Checker m_checker;
    mutable std::mutex m_mutex;
    mutable bool m_switched_to_reference = false;
};

// Sync infer request wrapper produced by AccuracyChecked::CompiledModel.
//
// On each infer() call it runs the main request, then copies its inputs
// to the reference request and runs that too.  Outputs from both are
// compared using the Checker provided at construction.  If any output
// fails the check the reference results are copied into the main output
// buffers and this request (and all subsequent requests from the same
// CompiledModel) permanently switch to reference-only inference.
class InferRequest final : public ov::ISyncInferRequest {
public:
    explicit InferRequest(std::shared_ptr<const CompiledModel> compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    void ensure_main_request_locked() const;
    void ensure_ref_request_locked() const;

    std::shared_ptr<const CompiledModel> m_acc_compiled_model;
    mutable std::mutex m_mutex;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_main_request;
    mutable std::shared_ptr<ov::IAsyncInferRequest> m_ref_request;
    mutable bool m_using_reference = false;
};

}  // namespace ov::npuw::accuracy_checked
