// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <optional>
#include <string>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

class DriverCompilerAdapter final : public ICompilerAdapter {
public:
    DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                    const FilteredConfig& config) const override;

    std::shared_ptr<IGraph> compileWS(std::shared_ptr<ov::Model>&& model, const FilteredConfig& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const FilteredConfig& config) const override;

    std::optional<std::vector<std::string>> get_supported_options() const override;

    bool is_option_supported(std::string optName, std::optional<std::string> optValue = std::nullopt) const override;

    uint32_t get_version() const override;

    bool validate_compatibility_descriptor(const std::string& compatibilityDescriptor) const override;

private:
    bool isCompilerOptionSupported(const FilteredConfig& config,
                                   const ze_graph_compiler_version_info_t& compilerVersion,
                                   const std::string& optionName) const;

    // Fetches the runtime requirements (compatibility descriptor) of a compiled graph from the
    // driver via zeDeviceGetRuntimeRequirements. Returns std::nullopt when the driver does not
    // implement the extension, the handle is null, or the query fails.
    std::optional<std::string> fetch_compatibility_descriptor(ze_graph_handle_t graphHandle) const;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;

    ze_device_graph_properties_t _compilerProperties = {};

    Logger _logger;
};

}  // namespace intel_npu
