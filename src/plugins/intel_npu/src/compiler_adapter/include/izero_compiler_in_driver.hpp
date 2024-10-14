// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/utils/logger/logger.hpp"
#include "npu.hpp"

namespace intel_npu {

class ILevelZeroCompilerInDriver {
public:
    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;
    virtual ze_graph_handle_t compile(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;
    virtual ze_graph_handle_t parse(const std::vector<uint8_t>& network, const Config& config) const = 0;

    virtual NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const = 0;

    virtual _ze_result_t release(ze_graph_handle_t graphHandle) = 0;

    virtual CompiledNetwork getCompiledNetwork(ze_graph_handle_t graphHandle) = 0;

    virtual void setArgumentValue(ze_graph_handle_t graphHandle, uint32_t argi_, const void* argv) const = 0;

    virtual void graphInitialie(ze_graph_handle_t graphHandle, const Config& config) const = 0;
};

}  // namespace intel_npu
