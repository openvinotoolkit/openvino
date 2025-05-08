// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.hpp"
#include "zero_pipeline.hpp"

namespace intel_npu {

class InitGraph : public Graph {
public:
    InitGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
              const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
              ze_graph_handle_t graphHandle,
              NetworkMetadata metadata,
              std::unique_ptr<BlobContainer> blobPtr,
              const Config& config,
              const ov::SoPtr<ICompiler>& compiler = {nullptr});

    InitInputData allocateInputs(const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants,
                                 const ov::SoPtr<ov::IRemoteContext>& context,
                                 const Config& config) override;

    InitOutputData allocateOutputs(const ov::SoPtr<ov::IRemoteContext>& context, const Config& config) override;

    void createPipeline(const Config& config,
                        const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
                        const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors);

    void runPipeline();

private:
    std::unique_ptr<Pipeline> _pipeline;
};

}  // namespace intel_npu
