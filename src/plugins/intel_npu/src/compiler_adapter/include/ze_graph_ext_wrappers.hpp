// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "vcl_serializer.hpp"

namespace intel_npu {

struct GraphDescriptor {
    GraphDescriptor(ze_graph_handle_t handle = nullptr, bool memoryPersistent = false);

    ze_graph_handle_t _handle = nullptr;
    bool _memoryPersistent = false;
};

/**
 * Adapter to use CiD through ZeroAPI
 */
class ZeGraphExtWrappers {
public:
    ZeGraphExtWrappers(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct);
    ZeGraphExtWrappers(const ZeGraphExtWrappers&) = delete;
    ZeGraphExtWrappers& operator=(const ZeGraphExtWrappers&) = delete;
    ~ZeGraphExtWrappers();

    std::unordered_set<std::string> queryGraph(SerializedIR serializedIR, const std::string& buildFlags) const;

    GraphDescriptor getGraphDescriptor(SerializedIR serializedIR,
                                       const std::string& buildFlags,
                                       const bool bypassUmdCache = false) const;

    GraphDescriptor getGraphDescriptor(const void* data, size_t size) const;

    NetworkMetadata getNetworkMeta(GraphDescriptor& graphDescriptor) const;

    void destroyGraph(GraphDescriptor& graphDescriptor);

    std::string getCompilerSupportedOptions() const;

    bool isOptionSupported(std::string optName, std::optional<std::string> optValue = std::nullopt) const;
    bool isTurboOptionSupported(const ze_graph_compiler_version_info_t& compilerVersion) const;

    void getGraphBinary(const GraphDescriptor& graphDescriptor,
                        std::vector<uint8_t>& blob,
                        const uint8_t*& blobPtr,
                        size_t& blobSize) const;

    void setGraphArgumentValue(const GraphDescriptor& graphDescriptor, uint32_t argi_, const void* argv) const;

    void initializeGraph(const GraphDescriptor& graphDescriptor, uint32_t commandQueueGroupOrdinal) const;

    bool isBlobDataImported(const GraphDescriptor& graphDescriptor) const;

private:
    void getMetadata(ze_graph_handle_t graphHandle,
                     uint32_t indexUsedByDriver,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    void initializeGraphThroughCommandList(ze_graph_handle_t graphHandle, uint32_t commandQueueGroupOrdinal) const;

    bool canCpuVaBeImported(const void* data, size_t size) const;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    uint32_t _graphExtVersion;

    Logger _logger;
};

// Parse the result string of query from format <name_0><name_1><name_2> to unordered_set of string
std::unordered_set<std::string> parseQueryResult(std::vector<char>& data);

}  // namespace intel_npu
