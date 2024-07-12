// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"
#include "openvino/runtime/properties.hpp"
#include "zero_init.hpp"
#include "zero_wrappers.hpp"

namespace intel_npu {

class ZeroExecutor final : public IExecutor {
public:
    ZeroExecutor(const std::shared_ptr<const ZeroInitStructsHolder>& initStructs,
                 const std::shared_ptr<const NetworkDescription>& networkDescription,
                 const Config& config,
                 const uint32_t& group_ordinal);

    ZeroExecutor(const ZeroExecutor&) = delete;
    ZeroExecutor& operator=(const ZeroExecutor&) = delete;

    ~ZeroExecutor() override;

    struct ArgumentDescriptor {
        ze_graph_argument_properties_3_t info;
        uint32_t idx;
    };

    void setArgumentValue(uint32_t argi_, const void* argv_) const;
    void setWorkloadType(const ov::WorkloadType workloadType) const override;
    inline ze_graph_handle_t graph() const {
        return _graph;
    }
    inline std::shared_ptr<const ZeroInitStructsHolder> getInitStructs() const {
        return _initStructs;
    }
    inline const std::shared_ptr<const NetworkDescription>& getNetworkDesc() const {
        return _networkDesc;
    }
    inline const std::array<std::shared_ptr<CommandQueue>, stage::COUNT>& getCommandQueue() const {
        return _command_queues;
    }
    inline const uint32_t& get_group_ordinal() const {
        return _group_ordinal;
    }
    inline const std::unordered_map<std::string, ArgumentDescriptor>& inputs_desc_map() const {
        return _inputs_desc_map;
    }
    inline const std::unordered_map<std::string, ArgumentDescriptor>& outputs_desc_map() const {
        return _outputs_desc_map;
    }

private:
    const Config _config;
    Logger _logger;

    const std::shared_ptr<const ZeroInitStructsHolder> _initStructs;
    std::shared_ptr<const NetworkDescription> _networkDesc;

    ze_graph_dditable_ext_curr_t* _graph_ddi_table_ext = nullptr;

    const uint32_t _group_ordinal;

    ze_graph_handle_t _graph = nullptr;
    ze_graph_properties_t _props{};

    std::unordered_map<std::string, ArgumentDescriptor> _inputs_desc_map;
    std::unordered_map<std::string, ArgumentDescriptor> _outputs_desc_map;

    std::array<std::shared_ptr<CommandQueue>, stage::COUNT> _command_queues;
};

}  // namespace intel_npu
