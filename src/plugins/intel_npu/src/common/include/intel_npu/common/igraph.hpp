// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"

namespace intel_npu {

/**
 * @struct CompiledNetwork
 * @brief Custom container for compiled network, used for export
 * @var CompiledNetwork::data
 * Pointer to the address of compiled network
 * @var CompiledNetwork:size
 * Size of the compiled network
 * @var CompiledNetwork::ownedStorage
 * Plugin owned compiled network storage that is required in case of a driver that
 * doesn't support graph extension 1.7, as in this case plugin must create a copy of the compiled network.
 * @note It's unsafe to store either data or size outside of the compiled network object as its destructor
 * would release the owning container
 */

struct CompiledNetwork {
    const uint8_t* data;
    size_t size;
    CompiledNetwork(const uint8_t* data, size_t size, std::vector<uint8_t> storage)
        : data(data),
          size(size),
          ownedStorage(std::move(storage)) {}

private:
    std::vector<uint8_t> ownedStorage;
};

class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph(ze_graph_handle_t handle, NetworkMetadata metadata) : _handle(handle), _metadata(std::move(metadata)) {}

    virtual CompiledNetwork export_blob() const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                                    const Config& config) const = 0;

    virtual void set_argument_value(uint32_t argi, const void* argv) const = 0;

    virtual void initialize(const Config& config) = 0;

    virtual ~IGraph() = default;

    const NetworkMetadata& get_metadata() const {
        return _metadata;
    }

    ze_graph_handle_t get_handle() const {
        return _handle;
    }

    void update_network_name(std::string_view name) {
        _metadata.name = name;
    }

    inline const std::vector<ArgumentDescriptor>& get_input_descriptors() const {
        return _input_descriptors;
    }

    inline const std::vector<ArgumentDescriptor>& get_output_descriptors() const {
        return _output_descriptors;
    }

    inline const std::shared_ptr<CommandQueue>& get_command_queue() const {
        return _command_queue;
    }

    void set_workload_type(const ov::WorkloadType workloadType) const {
        if (_command_queue == nullptr) {
            return;
        }

        ze_command_queue_workload_type_t zeWorkloadType;
        switch (workloadType) {
        case ov::WorkloadType::DEFAULT:
            zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT;
            break;
        case ov::WorkloadType::EFFICIENT:
            zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND;
            break;
        default:
            OPENVINO_THROW("Unknown value for WorkloadType!");
        }

        _command_queue->setWorkloadType(zeWorkloadType);
    }

    std::mutex& get_mutex() {
        return _mutex;
    }

protected:
    ze_graph_handle_t _handle = nullptr;
    NetworkMetadata _metadata;

    std::vector<ArgumentDescriptor> _input_descriptors;
    std::vector<ArgumentDescriptor> _output_descriptors;

    std::shared_ptr<CommandQueue> _command_queue;

    // Used to protect zero pipeline creation in the graph. The pipeline should be created only once per graph when the
    // first inference starts running
    std::mutex _mutex;
};

}  // namespace intel_npu
