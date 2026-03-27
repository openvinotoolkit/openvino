// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "graph.hpp"
#include "intel_npu/utils/zero/zero_tensor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace intel_npu {

/**
 * @brief Wrapper over multiple "ze_graph_handle_t" objects, one for each init/main schedule (weights separation).
 *
 * @details This class contains most implementation details for running the init schedules and setting the results as
 * inputs to the main one.
 */
class WeightlessGraph final : public Graph {
public:
    WeightlessGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                    const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                    const GraphDescriptor& mainGraphDesc,
                    NetworkMetadata mainMetadata,
                    std::optional<ov::Tensor> mainBlob,
                    const std::vector<GraphDescriptor>& initGraphDesc,
                    std::vector<NetworkMetadata> initMetadata,
                    std::optional<std::vector<ov::Tensor>> initBlobs,
                    std::shared_ptr<const ov::Model>&& model,
                    const FilteredConfig& config,
                    const bool blobIsPersistent = false);

    /**
     * @brief The main schedule along with the weights initialization ones are exported.
     */
    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream,
                                                                          const Config& config) override;

    /**
     * @brief Implementation hook for "IGraph::initialize" that initializes all underlying graph handles.
     * In addition to this, the init schedules are run and the result of this is set as inputs to the main
     * compiled model.
     */
    void initialize_impl(const FilteredConfig& config) override;

    // TODO: public for multi-threaded execution
    struct InputData {
        std::vector<std::shared_ptr<ov::ITensor>> tensors;
        std::shared_ptr<ZeroTensor> hostTensor;
    };

    struct OutputData {
        std::vector<std::shared_ptr<ov::ITensor>> tensors;
        std::shared_ptr<ZeroTensor> hostTensor;
        std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> tensorsMap;
    };

    ~WeightlessGraph();

private:
    /**
     * @brief Allocates the inputs of the init schedule using a single L0 buffer and copies the original weights there.
     *
     * @param initIndex Identifies the init schedule.
     * @param constants The original weights extracted from the ov::Model object.
     * @return The allocated L0 tensor along with view tensors corresponding to the init inputs.
     */
    InputData allocate_inputs(const size_t initIndex,
                              std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>& constants);

    /**
     * @brief Allocates the outputs of the init schedule using a single L0 buffer.
     *
     * @param initIndex Identifies the init schedule.
     * @return The allocated L0 tensor along with view tensors corresponding to the init outputs.
     */
    OutputData allocate_outputs(const size_t initIndex);

    /**
     * @brief Creates a pipeline for a single init schedule. This pipeline can be used for running inferences.
     */
    void create_pipeline(const size_t initIndex,
                         const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                         const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors);

    /**
     * @brief Runs the pipeline corresponding to a single init schedule.
     * @details This should produce the processed weights that can be used as inputs of the main schedule.
     */
    void run_pipeline(const size_t initIndex);

    void run_init_single_threaded();

    void run_init_multi_threaded();

    /**
     * @brief Sets the processed weights as inputs of the main schedule.
     */
    void set_weights_inputs();

    void release_init_blob(const size_t initIndex);
    void release_graphs();

    std::vector<GraphDescriptor> _initsGraphDesc;
    std::optional<std::vector<ov::Tensor>> _initBlobs;
    std::vector<NetworkMetadata> _initsMetadata;
    std::shared_ptr<const ov::Model> _model;

    std::vector<std::unique_ptr<CommandList>> _initsCommandLists;
    std::vector<std::unique_ptr<Fence>> _initsFences;
    std::shared_ptr<CommandQueue> _initsCommandQueue;

    /**
     * @brief Tensors holding the L0 buffers corresponding to the inputs of the main schedule.
     * @details Each vector entry corresponds to the output of one init schedule. The allocations have been performed
     * per init compiled model and not per init schedule output for performance reasons.
     */
    mutable std::vector<std::shared_ptr<ZeroTensor>> _mainInputsAllocatedTensors;
    /**
     * @brief Tensors pointing towards the buffers found in "_mainInputsAllocatedTensors".
     * @details Each map entry corresponds to one input of the main schedule.
     */
    mutable std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> _mainInputsViewTensors;
    Logger _wgLogger;  // Uses the "WeightlessGraph" domain
};

}  // namespace intel_npu
