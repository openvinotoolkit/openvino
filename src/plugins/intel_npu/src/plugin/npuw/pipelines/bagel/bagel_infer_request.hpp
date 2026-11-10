// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "bagel_compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

class BagelInferRequest : public ov::ISyncInferRequest {
public:
    explicit BagelInferRequest(std::shared_ptr<const BagelCompiledModel> compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void check_tensors() const override {}
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

private:
    // Inner request of one static variant, with its ports in the order of the outer model ports;
    // an output the variant does not compute (logits outside decode) is empty
    struct Variant {
        std::shared_ptr<ov::IAsyncInferRequest> request;
        std::vector<ov::Output<const ov::Node>> inputs;
        std::vector<std::optional<ov::Output<const ov::Node>>> outputs;
    };

    // Real and padded sizes of the last language-model call, used to crop its outputs
    struct LMCall {
        std::size_t seq = 0, seq_padded = 0, kv = 0, kv_padded = 0, und = 0, gen = 0;
    };

    Variant& variant_for(const ov::SoPtr<ov::ICompiledModel>& compiled);
    ov::SoPtr<ov::ITensor> input(const std::string& name) const;
    ov::SoPtr<ov::ITensor> inner_input(const Variant& variant, const std::string& name) const;
    void infer_language_model();
    void infer_per_shape();
    void crop_output(std::size_t idx);

    std::shared_ptr<const BagelCompiledModel> m_compiled_model;
    std::unordered_map<std::string, std::size_t> m_input_idx;
    std::map<const ov::ICompiledModel*, Variant> m_variants;

    Variant* m_last = nullptr;
    LMCall m_call;
    // Language-model outputs are cropped on first read, so unread outputs (e.g. the KV cache
    // during diffusion steps) cost nothing
    std::vector<bool> m_pending;
};

}  // namespace npuw
}  // namespace ov
