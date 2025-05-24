// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "graph.hpp"
#include "intel_npu/utils/zero/zero_host_tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace intel_npu {

class WeightlessGraph final : public Graph {
public:
    WeightlessGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                    const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                    const bool blobAllocatedByPlugin,
                    ze_graph_handle_t mainGraphHandle,
                    NetworkMetadata mainMetadata,
                    std::optional<ov::Tensor> mainBlob,
                    const std::vector<ze_graph_handle_t>& initGraphHandles,
                    std::vector<NetworkMetadata> initMetadata,
                    std::optional<std::vector<ov::Tensor>> initBlobs,
                    const std::shared_ptr<ov::Model>& model,
                    const Config& config,
                    const ov::SoPtr<ICompiler>& compiler = {nullptr});

    std::pair<uint64_t, std::vector<uint64_t>> export_blob(std::ostream& stream) const override;

    void initialize(const Config& config) override;

    void set_workload_type(const ov::WorkloadType workloadType) const override;

    // TODO: public for multi-threaded execution
    struct InputData {
        // TODO: is it necessary to keep both fields alive? it doesn't seem like
        // hostTensor field is ever used.
        std::vector<std::shared_ptr<ov::ITensor>> tensors;
        ov::SoPtr<ZeroHostTensor> hostTensor;
    };

    struct OutputData {
        // TODO: is it necessary to keep both fields alive? it doesn't seem like
        // hostTensor field is ever used.
        std::vector<std::shared_ptr<ov::ITensor>> tensors;
        ov::SoPtr<ZeroHostTensor> hostTensor;
        std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> tensorsMap;
    };

private:
    InputData allocate_inputs(const size_t initIndex,
                              const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants);

    OutputData allocate_outputs(const size_t initIndex);

    void create_pipeline(const size_t initIndex,
                         const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                         const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors);

    void run_pipeline(const size_t initIndex);

    /**
     * @brief TODO
     */
    void run_init_single_threaded();

    void run_init_multi_threaded();

    void set_weights_inputs();

    void free_init_resourcese(const size_t initIndex);

    std::vector<ze_graph_handle_t> _initHandles;
    std::optional<std::vector<ov::Tensor>> _initBlobs;
    std::vector<NetworkMetadata> _initMetadata;
    std::shared_ptr<ov::Model> _model;

    std::vector<std::vector<ArgumentDescriptor>> _initsInputDescriptors;
    std::vector<std::vector<ArgumentDescriptor>> _initsOutputDescriptors;

    std::vector<std::shared_ptr<CommandQueue>> _initsCommandQueues;
    std::vector<uint32_t> _initsCommandQueueOrdinals;
    std::vector<std::unique_ptr<CommandList>> _initsCommandLists;
    std::vector<std::unique_ptr<Fence>> _initsFences;

    /**
     * @brief TODO
     */
    mutable std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> _weightsInputs;
    mutable std::vector<ov::SoPtr<ZeroHostTensor>> _initOutputsTensors;
};

}  // namespace intel_npu
