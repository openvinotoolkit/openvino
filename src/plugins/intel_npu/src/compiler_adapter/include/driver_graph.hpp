// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "ze_graph_ext_wrapper_interface.hpp"
#include "zero_init.hpp"

namespace intel_npu {

class DriverGraph final : public IGraph {
public:
    DriverGraph(const std::shared_ptr<ZeGraphExtWrappersInterface>& zeGraphExt,
                const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                ze_graph_handle_t graphHandle,
                NetworkMetadata metadata,
                const Config& config,
                std::optional<std::vector<uint8_t>> network = std::nullopt);

    CompiledNetwork export_blob() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

    ~DriverGraph() override;

private:
    std::shared_ptr<ZeGraphExtWrappersInterface> _zeGraphExt;
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    Logger _logger;

    // We need to keep the compiled network inside the plugin when the model is imported.
    std::vector<uint8_t> _networkStorage;
};

}  // namespace intel_npu
