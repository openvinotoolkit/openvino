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
    InputData allocate_inputs(const size_t initIndex,
                              const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants);

    OutputData allocate_outputs(const size_t initIndex);

    void create_pipeline(const size_t initIndex);

    void run_pipeline(const size_t initIndex);

    /**
     * @brief TODO
     */
    void run_init_single_threaded();

    void run_init_multi_threaded();

    std::vector<ze_graph_handle_t> _initHandles;
    std::vector<NetworkMetadata> _initMetadata;
    std::vector<std::unique_ptr<BlobContainer>> _initBlobPtrs;

    std::vector<std::vector<ArgumentDescriptor>> _initsInputDescriptors;
    std::vector<std::vector<ArgumentDescriptor>> _initsOutputDescriptors;

    std::vector<std::shared_ptr<CommandQueue>> _initsCommandQueues;
    std::vector<uint32_t> _initsCommandQueueOrdinals;
    std::vector<std::unique_ptr<CommandList>> _initsCommandLists;
    std::vector<std::unique_ptr<Fence>> _initsFences;

    std::shared_ptr<ov::Model> _model;

    /**
     * @brief TODO
     */
    mutable std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> _weightsInputs;
    mutable std::vector<ov::SoPtr<ov::ITensor>> _initOutputsTensors;
};

}  // namespace intel_npu
