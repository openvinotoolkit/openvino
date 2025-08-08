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

class Graph : public IGraph {
public:
    Graph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
          const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
          const GraphDescriptor& graphDesc,
          NetworkMetadata metadata,
          std::optional<ov::Tensor> blob,
          const Config& config,
          const bool blobIsPersistent = false,
          const ov::SoPtr<ICompiler>& compiler = {nullptr},
          const bool calledFromWeightlessGraph = false);

    std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const override;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const Config& config) const override;

    void set_argument_value(uint32_t argi, const void* argv) const override;

    void initialize(const Config& config) override;

    const NetworkMetadata& get_metadata() const override;
    ze_graph_handle_t get_handle() const override;

    void update_network_name(std::string_view name) override;

    const std::vector<ArgumentDescriptor>& get_input_descriptors() const override;
    const std::vector<ArgumentDescriptor>& get_output_descriptors() const override;
    const std::shared_ptr<CommandQueue>& get_command_queue() const override;
    uint32_t get_command_queue_group_ordinal() const override;

    void set_workload_type(const ov::WorkloadType workloadType) const override;

    void set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) override;
    const std::shared_ptr<Event>& get_last_submitted_event(size_t indexOfCommandList) const override;
    void resize_last_submitted_event(size_t batch) override;
    void set_batch_size(std::size_t batch) override;
    void reset_last_batch_size() override;

    const std::optional<std::size_t> get_batch_size() const override;

    std::optional<size_t> determine_dynamic_batch_size(const std::shared_ptr<ov::ITensor>& tensor,
                                                       const std::optional<size_t> batchSize = std::nullopt,
                                                       const std::optional<size_t> index = std::nullopt,
                                                       const bool isInput = true) const override;

    uint32_t get_unique_id() override;
    void set_last_submitted_id(uint32_t id_index) override;
    uint32_t get_last_submitted_id() const override;

    ~Graph() override;

protected:
    bool release_blob(const Config& config);
    std::optional<size_t> determine_batch_size();

    std::shared_ptr<ZeGraphExtWrappers> _zeGraphExt;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;

    GraphDescriptor _graphDesc;
    NetworkMetadata _metadata;

    std::vector<ArgumentDescriptor> _inputDescriptors;
    std::vector<ArgumentDescriptor> _outputDescriptors;

    std::shared_ptr<CommandQueue> _commandQueue;
    uint32_t _commandQueueGroupOrdinal = 0;
    std::vector<std::shared_ptr<Event>> _lastSubmittedEvent;

    std::optional<ov::Tensor> _blob;

    // In the case of the import path, the blob is released after graph initialization so it can not be any longer
    // exported
    bool _blobIsReleased = false;
    bool _blobIsPersistent = false;

    uint32_t _uniqueId = 0;
    uint32_t _lastSubmittedId = 0;

    /**
     * @brief The batch size used by the corresponding model.
     * @details The attribute contains a value only if the plugin performs the batches splitting operation.
     */
    std::optional<std::size_t> _batchSize = std::nullopt;

    const ov::SoPtr<ICompiler> _compiler;
    Logger _logger;
};

}  // namespace intel_npu
