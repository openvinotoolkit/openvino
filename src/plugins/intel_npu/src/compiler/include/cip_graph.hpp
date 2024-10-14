// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "cid_compiler_adapter.hpp"
#include "igraph.hpp"
#include "npu.hpp"
#include "zero_backend.hpp"

namespace intel_npu {

class CipGraph final : public IGraph {
public:
    CipGraph(const std::shared_ptr<IEngineBackend>& iEngineBackend,
             NetworkMetadata metadata,
             std::vector<uint8_t> compiledNetwork,
             const Config& config);

    CompiledNetwork export_blob() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output() const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize() const override;

    ~CipGraph() override;

private:
    void initialize_graph_through_command_list() const;

    std::shared_ptr<ZeroEngineBackend> _zeroBackend;
    std::vector<uint8_t> _compiledNetwork;

    const Config _config;
    Logger _logger;
};

}  // namespace intel_npu
