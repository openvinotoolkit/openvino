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

class PluginGraph final : public IGraph {
public:
    PluginGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                const ov::SoPtr<ICompiler>& compiler,
                const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                ze_graph_handle_t graphHandle,
                NetworkMetadata metadata,
                std::unique_ptr<BlobContainer> blobPtr,
                const Config& config);

    size_t export_blob(std::ostream& stream) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

    ~PluginGraph() override;

private:
    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;
    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    const ov::SoPtr<ICompiler> _compiler;

    Logger _logger;
};

}  // namespace intel_npu
