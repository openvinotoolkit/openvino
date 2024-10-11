// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <mutex>

#include "igraph.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/properties.hpp"
#include "zero_init.hpp"
#include "zero_wrappers.hpp"

namespace intel_npu {

class ZeroExecutor final : public IExecutor {
public:
    ZeroExecutor(const std::shared_ptr<const ZeroInitStructsHolder>& initStructs,
                 const std::shared_ptr<IGraph>& graph,
                 const Config& config,
                 uint32_t group_ordinal);

    ZeroExecutor(const ZeroExecutor&) = delete;
    ZeroExecutor& operator=(const ZeroExecutor&) = delete;

    ~ZeroExecutor() override;

    struct ArgumentDescriptor {
        ze_graph_argument_properties_3_t info;
        uint32_t idx;
    };

    void setWorkloadType(const ov::WorkloadType workloadType) const override;
    void mutexLock() const;
    void mutexUnlock() const;

    inline const std::shared_ptr<CommandQueue>& getCommandQueue() const {
        return _command_queue;
    }
    inline const std::vector<ArgumentDescriptor>& get_input_descriptors() const {
        return _input_descriptors;
    }
    inline const std::vector<ArgumentDescriptor>& get_output_descriptors() const {
        return _output_descriptors;
    }

private:
    const Config _config;
    Logger _logger;

    const std::shared_ptr<const ZeroInitStructsHolder> _initStructs;
    std::shared_ptr<IGraph> _graph;

    std::vector<ArgumentDescriptor> _input_descriptors;
    std::vector<ArgumentDescriptor> _output_descriptors;

    std::shared_ptr<CommandQueue> _command_queue;

    mutable std::mutex _mutex;
};

}  // namespace intel_npu
