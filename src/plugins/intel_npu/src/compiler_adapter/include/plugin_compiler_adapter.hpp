// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <optional>

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "ivcl_compiler.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

class PluginCompilerAdapter final : public ICompilerAdapter {
public:
    /**
     * @brief Adapts an already-constructed compiler.
     *
     * The adapter never loads a compiler itself; composition is the caller's job. See
     * makeVCLCompiler() for the compiler-in-plugin.
     *
     * @param zeroInitStruct Pass null to construct without a Level Zero driver; the adapter then
     *        produces export-only graphs with no runtime metadata.
     * @param compiler The compiler-in-plugin to adapt; must be non-null.
     */
    PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                          ov::SoPtr<IVCLCompiler> compiler);

    std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                    const Config& config,
                                    const AdapterDescriptor& adapterDesc) const override;

    std::shared_ptr<IGraph> compileWS(std::shared_ptr<ov::Model>&& model,
                                      const Config& config,
                                      const AdapterDescriptor& adapterDesc) const override;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    std::vector<std::string> get_supported_options() const override;

    bool is_option_supported(const std::string& optName,
                             const std::optional<std::string>& optValue = std::nullopt) const override;

    uint32_t get_version() const override;

private:
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;
    ov::SoPtr<IVCLCompiler> _compiler;

    Logger _logger;
};

}  // namespace intel_npu
