// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include "intel_npu/network_metadata.hpp"

namespace intel_npu {

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;

class ZeGraphExtWrappersInterface {
public:
    virtual std::unordered_set<std::string> queryGraph(SerializedIR serializedIR,
                                                       const std::string& buildFlags) const = 0;

    virtual ze_graph_handle_t getGraphHandle(SerializedIR serializedIR,
                                             const std::string& buildFlags,
                                             const uint32_t& flags) const = 0;

    virtual ze_graph_handle_t getGraphHandle(const std::vector<uint8_t>& network) const = 0;

    virtual NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const = 0;

    virtual _ze_result_t destroyGraph(ze_graph_handle_t graphHandle) = 0;

    virtual void getGraphBinary(ze_graph_handle_t graphHandle,
                                std::vector<uint8_t>& blob,
                                const uint8_t*& blobPtr,
                                size_t& blobSize) const = 0;

    virtual void setGraphArgumentValue(ze_graph_handle_t graphHandle, uint32_t argi_, const void* argv) const = 0;

    virtual void initializeGraph(ze_graph_handle_t graphHandle, const Config& config) const = 0;

    virtual ~ZeGraphExtWrappersInterface() = default;
};

}  // namespace intel_npu
