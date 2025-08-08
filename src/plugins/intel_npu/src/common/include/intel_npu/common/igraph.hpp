// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph() = default;

    /**
     * @brief Writes the compiled model along with some metadata to the provided stream. The content of the stream can
     * later be used for importing the model.
     *
     * @param stream Where the content is placed
     * @return A pair made of the size of the main binary object and an optional variable. The optional variable
     * constitues the size of each init binary object if weights separation is enabled.
     */
    virtual std::pair<uint64_t, std::optional<std::vector<uint64_t>>> export_blob(std::ostream& stream) const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const Config& config) const = 0;

    virtual void set_argument_value(uint32_t argi, const void* argv) const = 0;

    virtual void initialize(const Config& config) = 0;

    virtual ~IGraph() = default;

    virtual const NetworkMetadata& get_metadata() const = 0;
    virtual ze_graph_handle_t get_handle() const = 0;

    virtual void update_network_name(std::string_view name) = 0;

    virtual const std::vector<ArgumentDescriptor>& get_input_descriptors() const = 0;
    virtual const std::vector<ArgumentDescriptor>& get_output_descriptors() const = 0;
    virtual const std::shared_ptr<CommandQueue>& get_command_queue() const = 0;
    virtual uint32_t get_command_queue_group_ordinal() const = 0;

    virtual void set_workload_type(const ov::WorkloadType workloadType) const = 0;

    std::mutex& get_mutex() {
        return _mutex;
    }

    virtual void set_last_submitted_event(const std::shared_ptr<Event>& event, size_t indexOfCommandList) = 0;
    virtual const std::shared_ptr<Event>& get_last_submitted_event(size_t indexOfCommandList) const = 0;
    virtual void resize_last_submitted_event(size_t batch) = 0;
    virtual void set_batch_size(std::size_t batch) = 0;

    virtual const std::optional<std::size_t> get_batch_size() const = 0;

    virtual std::optional<size_t> determine_dynamic_batch_size(const std::shared_ptr<ov::ITensor>& tensor,
                                                               const std::optional<size_t> batchSize = std::nullopt,
                                                               const std::optional<size_t> index = std::nullopt,
                                                               const bool isInput = true) const = 0;

    virtual uint32_t get_unique_id() = 0;
    virtual void set_last_submitted_id(uint32_t id_index) = 0;
    virtual uint32_t get_last_submitted_id() const = 0;

protected:
    // Used to protect zero pipeline creation in the graph. The pipeline should be created only once per graph when the
    // first inference starts running
    std::mutex _mutex;
};

}  // namespace intel_npu
