// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "../../compiled_model.hpp"
#include "../../perf.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov::npuw {

// Runs a model whose token count is dynamic (the Qwen-VL vision embeddings merger) on a static inner
// compiled model. Every dynamic input dimension is pinned to NPUW_VISION_MERGER_MAX_LEN, the public
// ports keep the model's dynamic shapes. A request pads the inputs up to the pinned length, hides
// the padding from the attention mask, and returns only the valid rows of the outputs.
class VisionMergerCompiledModel final : public ov::npuw::ICompiledModel {
public:
    // Pins the dynamic dimensions of `model` and compiles the result with the regular NPUW dispatch.
    VisionMergerCompiledModel(const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<const ov::IPlugin>& plugin,
                              const ov::AnyMap& properties);

    // Wraps an already compiled static `inner`; `model` supplies the dynamic public ports.
    VisionMergerCompiledModel(const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                              const std::shared_ptr<const ov::IPlugin>& plugin,
                              std::size_t max_len);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    void release_memory() override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    const std::shared_ptr<ov::npuw::ICompiledModel>& inner() const {
        return m_inner;
    }

    std::size_t max_len() const {
        return m_max_len;
    }

    // Indices of the dynamic dimensions of every port of the model, parallel to inputs() / outputs().
    const std::vector<std::vector<std::size_t>>& dynamic_input_dims() const {
        return m_in_dyn_dims;
    }
    const std::vector<std::vector<std::size_t>>& dynamic_output_dims() const {
        return m_out_dyn_dims;
    }

private:
    std::shared_ptr<ov::npuw::ICompiledModel> m_inner;
    std::size_t m_max_len = 0;
    std::vector<std::vector<std::size_t>> m_in_dyn_dims;
    std::vector<std::vector<std::size_t>> m_out_dyn_dims;
};

class VisionMergerInferRequest final : public ov::ISyncInferRequest {
public:
    VisionMergerInferRequest(const std::shared_ptr<const VisionMergerCompiledModel>& compiled_model,
                             std::shared_ptr<ov::IAsyncInferRequest> inner_request);

    void infer() override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // Token count shared by every dynamic input dimension.
    std::size_t valid_len() const;

    std::shared_ptr<const VisionMergerCompiledModel> m_model;
    std::shared_ptr<ov::IAsyncInferRequest> m_inner;
    std::mutex m_mutex;

    using MS = ov::npuw::perf::metric<ov::npuw::perf::MSec>;
    ov::npuw::perf::Profile<MS> m_profile;
};

}  // namespace ov::npuw
