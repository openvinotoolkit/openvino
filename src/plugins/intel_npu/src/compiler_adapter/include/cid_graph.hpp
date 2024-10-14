// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "igraph.hpp"
#include "izero_compiler_in_driver.hpp"
#include "npu.hpp"


namespace intel_npu {

class CidGraph final : public IGraph {
public:
    CidGraph(const std::shared_ptr<ILevelZeroCompilerInDriver>& apiAdapter,
             ze_graph_handle_t graphHandle,
             NetworkMetadata metadata,
             const Config& config);

    CompiledNetwork export_blob() const override;

    std::vector<ov::ProfilingInfo> process_profiling_output() const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize() const override;

    ~CidGraph() override;

private:
    std::shared_ptr<ILevelZeroCompilerInDriver> _apiAdapter;
    const Config _config;

    Logger _logger;
};

}  // namespace intel_npu
