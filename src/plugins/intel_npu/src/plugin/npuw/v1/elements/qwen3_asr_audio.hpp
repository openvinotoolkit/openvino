// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "../../compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov::npuw::qwen3_asr_audio {

class CompiledModel final : public ov::npuw::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& properties);

    static std::shared_ptr<CompiledModel> import_model(std::istream& stream,
                                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                                       const ov::AnyMap& properties);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;
    void release_memory() override;

    std::size_t window_frames() const;
    std::size_t mel_bins() const;
    std::size_t hidden_size() const;

private:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  std::shared_ptr<ov::npuw::ICompiledModel> inner,
                  std::size_t window_frames,
                  std::size_t mel_bins,
                  std::size_t hidden_size);
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    std::shared_ptr<ov::npuw::ICompiledModel> m_inner;
    std::size_t m_window_frames = 0;
    std::size_t m_mel_bins = 0;
    std::size_t m_hidden_size = 0;
};

class InferRequest final : public ov::ISyncInferRequest {
public:
    InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model,
                 std::shared_ptr<ov::IAsyncInferRequest> inner);

    void infer() override;
    void check_tensors() const override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    std::shared_ptr<const CompiledModel> m_model;
    std::shared_ptr<ov::IAsyncInferRequest> m_inner;
};

}  // namespace ov::npuw::qwen3_asr_audio
