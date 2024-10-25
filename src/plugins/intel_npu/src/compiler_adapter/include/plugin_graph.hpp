// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/icompiler.hpp"
#include "izero_adapter.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class PluginGraph final : public IGraph {
public:
    PluginGraph(const std::shared_ptr<IZeroAdapter>& adapter,
                const ov::SoPtr<ICompiler>& compiler,
                ze_graph_handle_t graphHandle,
                NetworkMetadata metadata,
                std::vector<uint8_t> compiledNetwork,
                const Config& config);

    CompiledNetwork export_blob() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize() override;

    ~PluginGraph() override;

private:
    std::shared_ptr<IZeroAdapter> _adapter;
    const ov::SoPtr<ICompiler> _compiler;
    std::vector<uint8_t> _compiledNetwork;

    const Config _config;
    Logger _logger;
};

}  // namespace intel_npu
