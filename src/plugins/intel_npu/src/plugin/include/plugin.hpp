// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "backends_registry.hpp"
#include "blob_format_importers.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
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
    void update_log_level(const ov::AnyMap& properties) const;

    /**
     * @brief Looks for "DISABLE_VERSION_CHECK" and "IMPORT_RAW_BLOB" to determine whether or not the blob to be
     * imported should be treated as a "raw" one (i.e. the whole blob is a compiler main schedule).
     */
    bool should_import_raw_blob(const ov::AnyMap& properties) const;

    std::shared_ptr<ov::ICompiledModel> import_model(const std::unique_ptr<IBlobFormatImporter>& blobFormatHandler,
                                                     FilteredConfig& localConfig,
                                                     ov::AnyMap& localProperties) const;

    std::unique_ptr<BackendsRegistry> _backendsRegistry;

    //  _backend might not be set by the plugin; certain actions, such as offline compilation, might be supported.
    //  Appropriate checks are needed in plugin/metrics/properties when actions depend on a backend.
    ov::SoPtr<IEngineBackend> _backend;

    mutable Logger _logger;
    std::unique_ptr<PluginPropertyManager> _propertiesManager;

    static std::atomic<int> _compiledModelLoadCounter;
};

}  // namespace intel_npu
