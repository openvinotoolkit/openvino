// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "npuw/compiled_model.hpp"

namespace ov::npuw {

// The front-end for the dynamic, stateless PagedAttention model deployed by
// the GenAI continuous-batching pipeline. This class settles the plugin-level
// seams: the model is compiled 1:1 on the PA fallback device (CPU unless the
// internal OPENVINO_NPUW_PA_DEVICE env var says otherwise), the
// NPU*-prefixed properties are held at this level while everything else is
// forwarded to the executing device, and the device-resolved KV cache
// geometry is stamped back onto the exposed ports so the pipeline's
// KVCacheManager reads the real cache precision and block shape.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    // The inner model is compiled first and the resolved KV cache element
    // types / shapes are stamped back onto the model's cache Parameters, so
    // this compiled model's ports expose the real geometry. The CB pipeline's
    // KVCacheManager reads precision and block shape off these ports.
    struct PreparedState {
        std::shared_ptr<ov::Model> model;
        ov::SoPtr<ov::ICompiledModel> compiled;
        std::string device;
    };
    static PreparedState prepare(const std::shared_ptr<ov::Model>& model,
                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                 const ov::AnyMap& properties);

    PACompiledModel(PreparedState prepared, const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    std::string m_device;
    ov::SoPtr<ov::ICompiledModel> m_compiled_model;

    // KV cache block size as fixed by the device at compile time; 0 if the
    // compiled cache shape is still dynamic in that dimension.
    std::size_t m_block_size = 0u;
};

}  // namespace ov::npuw
