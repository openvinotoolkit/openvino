// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include <memory>
#include <vector>

#include "intel_npu/al/icompiler.hpp"

namespace intel_npu {

class IExecutor {
public:
    virtual ~IExecutor() = default;

    virtual void setWorkloadType(const ov::WorkloadType workloadType) const = 0;
};

class IGraph : public std::enable_shared_from_this<IGraph> {
public:
    IGraph(ze_graph_handle_t handle, NetworkMetadata metadata) : _handle(handle), _metadata(std::move(metadata)) {}

    virtual CompiledNetwork export_blob() const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData) const = 0;

    virtual void set_argument_value(uint32_t argi, const void* argv) const = 0;

    virtual void initialize() = 0;

    virtual ~IGraph() = default;

    struct ArgumentDescriptor {
        ze_graph_argument_properties_3_t info;
        uint32_t idx;
    };

    void setWorkloadType(const ov::WorkloadType workloadType) const {
        if (_executor != nullptr) {
            _executor->setWorkloadType(workloadType);
        }
    }

    const std::shared_ptr<IExecutor>& get_executor() const {
        return _executor;
    }

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

protected:
    ze_graph_handle_t _handle = nullptr;
    NetworkMetadata _metadata;

    std::shared_ptr<IExecutor> _executor;

    std::vector<ArgumentDescriptor> _input_descriptors;
    std::vector<ArgumentDescriptor> _output_descriptors;
};

}  // namespace intel_npu
