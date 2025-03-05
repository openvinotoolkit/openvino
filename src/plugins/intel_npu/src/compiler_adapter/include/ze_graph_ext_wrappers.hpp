// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;

/**
 * Adapter to use CiD through ZeroAPI
 */
class ZeGraphExtWrappers {
public:
    ZeGraphExtWrappers(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);
    ZeGraphExtWrappers(const ZeGraphExtWrappers&) = delete;
    ZeGraphExtWrappers& operator=(const ZeGraphExtWrappers&) = delete;
    ~ZeGraphExtWrappers();

    std::unordered_set<std::string> queryGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                               const std::string& buildFlags) const;
    ze_graph_handle_t getGraphHandle(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                     const std::string& buildFlags,
                                     const uint32_t& flags) const;

    ze_graph_handle_t getGraphHandle(const uint8_t& data, size_t size) const;

    NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const;

    _ze_result_t destroyGraph(ze_graph_handle_t graphHandle);

    void getGraphBinary(ze_graph_handle_t graphHandle,
                        std::vector<uint8_t>& blob,
                        const uint8_t*& blobPtr,
                        size_t& blobSize) const;

    void setGraphArgumentValue(ze_graph_handle_t graphHandle, uint32_t argi_, const void* argv) const;

    void initializeGraph(ze_graph_handle_t graphHandle, uint32_t commandQueueGroupOrdinal) const;

private:
    std::unordered_set<std::string> getQueryResultFromSupportedLayers(
        ze_result_t result,
        ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    void getMetadata(ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    void initialize_graph_through_command_list(ze_graph_handle_t graphHandle, uint32_t commandQueueGroupOrdinal) const;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    uint32_t _graphExtVersion;

    Logger _logger;
};

}  // namespace intel_npu
