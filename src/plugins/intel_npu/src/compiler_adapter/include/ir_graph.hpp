// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "ze_graph_ext_wrappers.hpp"

namespace intel_npu {
class IRGraph : public intel_npu::IGraph {
public:
    IRGraph(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
            const Config& config,
            NetworkMetadata metadata,
            std::unique_ptr<BlobContainer> blobPtr);

    //~IRGraph() {}

    size_t export_blob(std::ostream&) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>&, const Config&) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

private:
    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    std::vector<uint8_t> _blob;
    Logger _logger;
};
}  // namespace intel_npu
