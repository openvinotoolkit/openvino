// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/common/blob_container.hpp"
#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph(ze_graph_handle_t handle,
           NetworkMetadata metadata,
           const Config& config,
           std::unique_ptr<BlobContainer> blobPtr);

    virtual size_t export_blob(std::ostream& stream) const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const Config& config) const = 0;

    virtual void set_argument_value(uint32_t argi, const void* argv) const = 0;

    virtual void initialize(const Config& config) = 0;

    virtual ~IGraph() = default;

    const NetworkMetadata& get_metadata() const;
    ze_graph_handle_t get_handle() const;

    void update_network_name(std::string_view name);

    const std::vector<ArgumentDescriptor>& get_input_descriptors() const;
    const std::vector<ArgumentDescriptor>& get_output_descriptors() const;
    const std::shared_ptr<CommandQueue>& get_command_queue() const;
    uint32_t get_command_queue_group_ordinal() const;

    void set_workload_type(const ov::WorkloadType workloadType) const;

    std::mutex& get_mutex();

    void set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList);
    const std::shared_ptr<Event>& get_last_submitted_event(size_t indexOfCommandList) const;

    uint32_t get_unique_id();
    void set_last_submitted_id(uint32_t id_index);
    uint32_t get_last_submitted_id() const;

    const std::optional<std::size_t> get_batch_size() const;

protected:
    /**
     * @brief Determines if batching can be addressed inside the plugin. In the positive case, the batch size used by
     * the model will also be deduced and returned.
     * @details Batching can be handled by the plugin only if:
     *  - The batch axis is the first axis.
     *  - The batch size received by the compiler takes the default value of 1.
     *  - The batch size found in the IR model matches for all inputs/outputs and takes a value different than the
     * default one.
     *
     * If any of the previous conditions is not fulfilled, the functon will return the default batch size, thus no
     * custom algorithm will be applied inside the plugin in order to address batching.
     *
     * @param metadata Metadata containing the shape values as seen by both the compiler and IR model. These will
     * ultimately be used for determining the batch size.
     * @returns The batch size deduced by the algorithm or the default value of 1 if batching cannot be performed inside
     * the plugin.
     */
    std::optional<size_t> get_batch_size(const NetworkMetadata& metadata);

    ze_graph_handle_t _handle = nullptr;
    NetworkMetadata _metadata;

    std::vector<ArgumentDescriptor> _input_descriptors;
    std::vector<ArgumentDescriptor> _output_descriptors;

    std::shared_ptr<CommandQueue> _command_queue;
    uint32_t _command_queue_group_ordinal = 0;
    std::vector<std::shared_ptr<Event>> _last_submitted_event;

    // Used to protect zero pipeline creation in the graph. The pipeline should be created only once per graph when the
    // first inference starts running
    std::mutex _mutex;

    std::unique_ptr<BlobContainer> _blobPtr;

    uint32_t _unique_id = 0;
    uint32_t _last_submitted_id = 0;

    /**
     * @brief The batch size used by the corresponding model.
     * @details The attribute contains a value only if the plugin performs the batches splitting operation.
     */
    std::optional<std::size_t> _batch_size = std::nullopt;

    Logger _logger;
};

}  // namespace intel_npu
