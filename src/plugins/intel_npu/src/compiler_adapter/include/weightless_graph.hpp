// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "graph.hpp"

namespace intel_npu {

class WeightlessGraph final : public Graph {
public:
    WeightlessGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                    const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                    ze_graph_handle_t mainGraphHandle,
                    NetworkMetadata mainMetadata,
                    std::unique_ptr<BlobContainer> mainBlobPtr,
                    const std::vector<ze_graph_handle_t>& initGraphHandles,
                    const std::vector<NetworkMetadata>& initMetadata,
                    const std::vector<std::unique_ptr<BlobContainer>>& initBlobPtrs,
                    const Config& config,
                    const ov::SoPtr<ICompiler>& compiler = {nullptr});

    std::pair<uint64_t, std::vector<uint64_t>> export_blob(std::ostream& stream) const override;

    void initialize(const Config& config) override;

    ~WeightlessGraph() override;

private:
    std::vector<ze_graph_handle_t> _initHandles;
    std::vector<NetworkMetadata> _initMetadata;
    std::vector<std::unique_ptr<BlobContainer>> _initBlobPtrs;
};

}  // namespace intel_npu
