// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "graph.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iremote_context.hpp"

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
                    const std::shared_ptr<ov::Model>& model,
                    const Config& config,
                    const ov::SoPtr<ICompiler>& compiler = {nullptr});

    std::pair<uint64_t, std::vector<uint64_t>> export_blob(std::ostream& stream) const override;

    void initialize(const Config& config) override;

    ~WeightlessGraph() override;

    // TODO: public for multi-threaded execution
    struct InputData {
        // TODO: is it necessary to keep both fields alive? it doesn't seem like
        // hostTensor field is ever used.
        std::vector<std::vector<std::shared_ptr<ov::ITensor>>> tensors;
        ov::SoPtr<ov::ITensor> hostTensor;
    };

    struct OutputData {
        // TODO: is it necessary to keep both fields alive? it doesn't seem like
        // hostTensor field is ever used.
        std::vector<std::shared_ptr<ov::ITensor>> tensors;
        ov::SoPtr<ov::ITensor> hostTensor;
        std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> tensorsMap;
    };

private:
    InputData allocateInputs(const std::shared_ptr<IGraph>& initGraph,
                             const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants,
                             const ov::SoPtr<ov::IRemoteContext>& context,
                             const Config& config);

    OutputData allocateOutputs(const std::shared_ptr<IGraph>& initGraph,
                               const ov::SoPtr<ov::IRemoteContext>& context,
                               const Config& config);

    /**
     * @brief TODO
     */
    std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, ov::SoPtr<ov::ITensor>> runInit(
        const std::shared_ptr<IGraph>& initGraph,
        const std::shared_ptr<const ov::Model>& model,
        const ov::SoPtr<ov::IRemoteContext>& context,
        const Config& config) override;

    std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, std::vector<ov::SoPtr<ov::ITensor>>>
    runInitMultiThreaded(const std::vector<std::shared_ptr<IGraph>>& initGraph,
                         const std::shared_ptr<const ov::Model>& model,
                         const ov::SoPtr<ov::IRemoteContext>& context,
                         const Config& config) override;

    std::vector<ze_graph_handle_t> _initHandles;
    std::vector<NetworkMetadata> _initMetadata;
    std::vector<std::unique_ptr<BlobContainer>> _initBlobPtrs;

    std::vector<std::vector<ArgumentDescriptor>> _initsInputDescriptors;
    std::vector<std::vector<ArgumentDescriptor>> _initsOutputDescriptors;

    std::vector<std::shared_ptr<CommandQueue>> _initsCommandQueues;
    std::vector<uint32_t> _initsCommandQueueOrdinals;

    std::shared_ptr<ov::Model> _model;
    ov::SoPtr<ov::IRemoteContext> _remoteContext;
};

}  // namespace intel_npu
