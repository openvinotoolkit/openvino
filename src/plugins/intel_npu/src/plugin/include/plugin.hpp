// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "backends_registry.hpp"
#include "blob_source.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "plugin_property_manager.hpp"

namespace intel_npu {

class Plugin : public ov::IPlugin {
public:
    Plugin();

    Plugin(const Plugin&) = delete;

    Plugin& operator=(const Plugin&) = delete;

    ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    bool is_property_supported(const std::string& name, const ov::AnyMap& arguments = {}) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remoteProperties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remoteProperties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& compiledBlob,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& compiledBlob,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    void update_global_properties(const ov::AnyMap& properties) const;

    std::shared_ptr<ov::ICompiledModel> import_model(BlobSource& blobSource, ov::AnyMap& properties) const;

    /**
     * @brief Returns a provider of the VCL entry points for the components that need them.
     *
     * The provider captures the loader holder, not `this`, so it stays valid when a longer lived
     * component keeps it. The library loads on the first call and stays loaded while any provider
     * or returned table still owns it.
     */
    VCLFunctionTableProvider vcl_functions() const {
        return [holder = _vclLoaderHolder] {
            return holder->functions();
        };
    }

    std::unique_ptr<BackendsRegistry> _backendsRegistry;

    //  _backend might not be set by the plugin; certain actions, such as offline compilation, might be supported.
    //  Appropriate checks are needed in plugin/metrics/properties when actions depend on a backend.
    ov::SoPtr<IEngineBackend> _backend;

    // Defers VCL loading until compiler construction or profiling decoder creation needs it.
    std::shared_ptr<VCLLoaderHolder> _vclLoaderHolder;

    // PREFER_PLUGIN probes compiler availability and remembers whether to use the plugin or driver compiler.
    // A process-wide cache can reuse another provider's result, choosing DRIVER despite an available compiler
    // or PLUGIN despite an unavailable one. Keep one factory per plugin so compile/query operations,
    // property queries and option support checks share the result for this plugin's VCL provider.
    // Creating a fresh factory for each operation would lose that result and repeat failed compiler probes.
    // Factory copies held by the helpers share this cache; explicit PLUGIN requests still retry after failure.
    std::unique_ptr<CompilerAdapterFactory> _compilerAdapterFactory;

    mutable Logger _logger;
    std::unique_ptr<PluginPropertyManager> _propertiesManager;
    std::shared_ptr<CompilerOptionSupportHelper> _compilerOptionSupportHelper;

    static std::atomic<int> _compiledModelLoadCounter;
};

}  // namespace intel_npu
