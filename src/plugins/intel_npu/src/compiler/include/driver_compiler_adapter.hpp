// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "backends.hpp"
#include "iexternal_compiler.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "zero_types.hpp"
namespace intel_npu {
namespace driverCompilerAdapter {

/**
 * @brief Adapter for Compiler in driver
 * @details Wrap compiler in driver calls and do preliminary actions (like opset conversion)
 */
class LevelZeroCompilerAdapter final : public ICompiler {
public:
    LevelZeroCompilerAdapter(std::shared_ptr<NPUBackends> npuBackends);

    uint32_t getSupportedOpsetVersion() const override final;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                               const Config& config) const override final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override final;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const override final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const Config& config) const override final;

    void releaseGraphHandle(void* graphHandle);

    void getCompiledNetwork(void* graphHandle, std::vector<uint8_t>& compiledNetwork);

    std::pair<NetworkDescription, void*> compileAndReturnGraph(const std::shared_ptr<const ov::Model>& model,
                                                               const Config& config);

    std::pair<NetworkMetadata, void*> parseAndReturnGraph(const std::vector<uint8_t>& network, const Config& config);

private:
    /**
     * @brief Separate externals calls to separate class
     */
    std::shared_ptr<IExternalCompiler> apiAdapter;
    Logger _logger;
};

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
