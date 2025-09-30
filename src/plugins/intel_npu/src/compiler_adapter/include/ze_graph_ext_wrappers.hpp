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

    std::unordered_set<std::string> queryGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                               const std::string& buildFlags) const;

    GraphDescriptor getGraphDescriptor(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                       const std::string& buildFlags,
                                       const uint32_t& flags) const;

    GraphDescriptor getGraphDescriptor(void* data, size_t size) const;

    NetworkMetadata getNetworkMeta(GraphDescriptor& graphDescriptor) const;

    void destroyGraph(GraphDescriptor& graphDescriptor);

    std::string getCompilerSupportedOptions() const;

    bool isOptionSupported(std::string optname) const;
    bool isTurboOptionSupported(const ze_graph_compiler_version_info_t& compilerVersion) const;

    void getGraphBinary(const GraphDescriptor& graphDescriptor,
                        std::vector<uint8_t>& blob,
                        const uint8_t*& blobPtr,
                        size_t& blobSize) const;

    void setGraphArgumentValue(const GraphDescriptor& graphDescriptor, uint32_t argi_, const void* argv) const;

    void initializeGraph(const GraphDescriptor& graphDescriptor, uint32_t commandQueueGroupOrdinal) const;

private:
    void getMetadata(ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    void initializeGraphThroughCommandList(ze_graph_handle_t graphHandle, uint32_t commandQueueGroupOrdinal) const;

    bool canCpuVaBeImported(void* data, size_t size, const uint32_t flags = 0) const;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    uint32_t _graphExtVersion;

    Logger _logger;
};

}  // namespace intel_npu
