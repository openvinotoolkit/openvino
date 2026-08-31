// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <optional>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "ivcl_compiler.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

class PluginCompilerAdapter final : public ICompilerAdapter {
public:
    /**
     * @brief Loads the compiler-in-plugin and adapts it. Production entry point.
     */
    PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                          const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr,
                          const std::optional<IDevice::DeviceProperties>& deviceProperties = std::nullopt);

    /**
     * @brief Adapts an already-constructed compiler.
     *
     * Injecting the compiler is what makes the adapter's own logic - blob-type dispatch, the
     * weights-separation main/init split, the option-support cache interplay - testable without a
     * driver or an NPU. Pass a null @p zeroInitStruct to exercise the no-driver path.
     */
    PluginCompilerAdapter(ov::SoPtr<IVCLCompiler> compiler,
                          const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                          const std::shared_ptr<OptionSupportCache>& optionSupportCache = nullptr);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                    const FilteredConfig& config) const override;

    std::shared_ptr<IGraph> compileWS(std::shared_ptr<ov::Model>&& model, const FilteredConfig& config) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const FilteredConfig& config) const override;

    std::vector<std::string> get_supported_options() const override;

    bool is_option_supported(const std::string& optName,
                             const std::optional<std::string>& optValue = std::nullopt) const override;

    uint32_t get_version() const override;

private:
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    std::shared_ptr<OptionSupportCache> _optionSupportCache;
    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;
    ov::SoPtr<IVCLCompiler> _compiler;

    Logger _logger;
};

}  // namespace intel_npu
