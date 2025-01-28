// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {

class DriverGraph final : public IGraph {
public:
    DriverGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                ze_graph_handle_t graphHandle,
                NetworkMetadata metadata,
                const Config& config,
                std::unique_ptr<BlobContainer> blobPtr);

    size_t export_blob(std::ostream& stream) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

    ~DriverGraph() override;

private:
    bool release_blob(const Config& config);

    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    Logger _logger;

    // In the case of the import path, the blob is released after graph initialization so it can not be any longer
    // exported
    bool _blobIsReleased = false;
};

}  // namespace intel_npu
