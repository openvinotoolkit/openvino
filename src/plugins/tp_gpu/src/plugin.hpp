// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/iplugin.hpp"
#include "tp_l0_shared_context.hpp"

namespace ov {
namespace tp_gpu {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

private:
    /// One Level Zero context spanning every rank device, plus the per-rank
    /// intel_gpu remote contexts that adopt it.
    struct SharedL0Setup {
        TPL0SharedContextPtr shared;
        std::vector<ov::SoPtr<ov::IRemoteContext>> rank_ctx;
    };

    /// Builds the shared context. Used by both compile and import: an
    /// imported model needs exactly the same cross-device context to run
    /// collectives on, and there is nothing in the blob that could replace it.
    SharedL0Setup create_shared_l0(const std::vector<std::string>& device_names) const;

    /// Restores a compiled model from a TP blob. Shared by all four
    /// `import_model` overloads; the tensor ones only wrap the memory in a
    /// stream first.
    std::shared_ptr<ov::ICompiledModel> import_blob(std::istream& blob, const ov::AnyMap& properties) const;

    /// Best-effort rank device list for property queries, which arrive without
    /// a compile config. Falls back to the GPU plugin's own default device.
    std::vector<std::string> query_devices(const ov::AnyMap& arguments) const;

    /// The GPU plugin's cache-defining properties, which are also ours: a TP
    /// blob is a container of intel_gpu blobs.
    std::vector<ov::PropertyName> gpu_caching_properties(const ov::AnyMap& arguments) const;

    /// Answers a GPU-owned property on behalf of the whole rank set.
    ov::Any aggregate_rank_property(const std::string& name, const ov::AnyMap& arguments) const;

    mutable ov::AnyMap m_config;
};

}  // namespace tp_gpu
}  // namespace ov
