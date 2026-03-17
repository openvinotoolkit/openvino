// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "model_serializer.hpp"

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

    /**
     * @brief Returns the list of compiler options supported by the driver.
     * @return `std::optional<std::string>` containing the list of supported options if the query is supported,
     *         or `std::nullopt` if the query itself is not supported.
     */
    std::optional<std::string> getCompilerSupportedOptions() const;

    /**
     * @brief Checks whether the specified driver/compiler option is supported by the driver.
     * @param optName The name of the option to check.
     * @param optValue The value of the option to check (optional).
     * @return `true` if the option is supported, `false` if it is not supported,
     *         and `std::nullopt` if the option-support query itself is not supported.
     */
    std::optional<bool> isOptionSupported(std::string optName,
                                          std::optional<std::string> optValue = std::nullopt) const;

    /**
     * @brief Tells us whether or not the driver is able to receive and take into account a hash of the model instead of
     * computing its own within the UMD.
     */
    bool isPluginModelHashSupported() const;

    void getGraphBinary(const GraphDescriptor& graphDescriptor,
                        std::vector<uint8_t>& blob,
                        const uint8_t*& blobPtr,
                        size_t& blobSize) const;

    void setGraphArgumentValue(const GraphDescriptor& graphDescriptor, uint32_t id, const void* data) const;

    void setGraphArgumentValueWithStrides(const GraphDescriptor& graphDescriptor,
                                          uint32_t id,
                                          const void* data,
                                          const std::vector<size_t>& strides) const;

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
    bool _isCompilerOptionQuerySupported;

    Logger _logger;
};

// Parse the result string of query from format <name_0><name_1><name_2> to unordered_set of string
std::unordered_set<std::string> parseQueryResult(std::vector<char>& data);

}  // namespace intel_npu
