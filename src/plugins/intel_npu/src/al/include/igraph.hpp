// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {
class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph(ze_graph_handle_t handle, NetworkMetadata metadata) : _handle(handle), _metadata(std::move(metadata)) {}

    virtual CompiledNetwork export_blob() const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData) const = 0;

    virtual void set_argument_value(uint32_t argi, const void* argv) const = 0;

    virtual void initialize() = 0;

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

    void setWorkloadType(const ov::WorkloadType workloadType) const {
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

    void mutexLock() {
        _mutex.lock();
    }

    void mutexUnlock() {
        _mutex.unlock();
    }

protected:
    ze_graph_handle_t _handle = nullptr;
    NetworkMetadata _metadata;

    std::vector<ArgumentDescriptor> _input_descriptors;
    std::vector<ArgumentDescriptor> _output_descriptors;

    std::shared_ptr<CommandQueue> _command_queue;

    std::mutex _mutex;
};

}  // namespace intel_npu
