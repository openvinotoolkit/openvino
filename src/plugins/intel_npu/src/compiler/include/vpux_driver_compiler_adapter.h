//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "iexternal_compiler.h"
#include "npu/utils/logger/logger.hpp"
#include "vpux/al/icompiler.hpp"

using intel_npu::Config;
using intel_npu::Logger;

namespace vpux {
namespace driverCompilerAdapter {

/**
 * @brief Adapter for Compiler in driver
 * @details Wrap compiler in driver calls and do preliminary actions (like opset conversion)
 */
class LevelZeroCompilerAdapter final : public ICompiler {
public:
    LevelZeroCompilerAdapter();

    uint32_t getSupportedOpsetVersion() const final;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const final;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const Config& config) const final;

private:
    /**
     * @brief Separate externals calls to separate class
     */
    std::shared_ptr<IExternalCompiler> apiAdapter;
    ze_driver_handle_t _driverHandle = nullptr;
    mutable Logger _logger;
};

}  // namespace driverCompilerAdapter
}  // namespace vpux
