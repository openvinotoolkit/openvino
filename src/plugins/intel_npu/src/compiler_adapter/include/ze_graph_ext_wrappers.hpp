// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_graph_ext.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "intel_npu/common/network_metadata.hpp"
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
                                       const bool bypassUmdCache = false,
                                       const bool secureCompile = false) const;

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
    std::optional<bool> isOptionSupported(const std::string& optName,
                                          const std::optional<std::string>& optValue = std::nullopt) const;

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

    void initializeGraph(const GraphDescriptor& graphDescriptor) const;

    bool isBlobDataImported(const GraphDescriptor& graphDescriptor) const;

    /**
     * @brief Tells whether the graph is known to require the initialize stage.
     * @return `true` only when the driver states the requirement. Anything that leaves it
     *         unestablished - no graph handle, a driver too old to express it, or a failed
     *         query - reports `false`.
     */
    bool isInitStageRequired(const GraphDescriptor& graphDescriptor) const;

    /**
     * @brief Tells whether the graph was compiled with layer profiling enabled.
     * @return `std::nullopt` if the driver is too old to answer.
     */
    std::optional<bool> isProfilingEnabled(const GraphDescriptor& graphDescriptor) const;

    void evict_memory(const GraphDescriptor& graphDescriptor) const;

    /**
     * @brief Fetches the compatibility descriptor of a compiled graph.
     * @return The descriptor string if the driver supports the feature and the graph carries one,
     *         or `std::nullopt` otherwise.
     */
    std::optional<std::string> getCompatibilityDescriptor(ze_graph_handle_t graphHandle) const;

private:
    void getMetadata(ze_graph_handle_t graphHandle,
                     uint32_t indexUsedByDriver,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    /**
     * @brief Adds the descriptor of an argument queried through "pfnGetArgumentProperties4" (graph ext 1.20+).
     * @param name The name of the argument, as already queried for finding its IR metadata.
     * @param metadata The IR metadata of the argument, from which the rank is taken.
     */
    void getMetadataFromProperties4(ze_graph_handle_t graphHandle,
                                    uint32_t indexUsedByDriver,
                                    std::string name,
                                    const ze_graph_argument_metadata_t& metadata,
                                    std::vector<IODescriptor>& inputs,
                                    std::vector<IODescriptor>& outputs) const;

    /**
     * @brief Adds the descriptor of an argument queried through "pfnGetArgumentProperties3", which is limited to 5
     * dimensions.
     * @param readTensorNames Whether to read the associated tensor names from the fixed-size array of the structure.
     * They are left empty otherwise.
     */
    void getMetadataFromProperties3(ze_graph_handle_t graphHandle,
                                    uint32_t indexUsedByDriver,
                                    bool readTensorNames,
                                    std::vector<IODescriptor>& inputs,
                                    std::vector<IODescriptor>& outputs) const;

    /**
     * @brief Queries the whole answer of "pfnGetArgumentNames" (graph ext 1.20+), whatever its content.
     */
    std::vector<char> queryArgumentNames(ze_graph_handle_t graphHandle,
                                         uint32_t indexUsedByDriver,
                                         ze_graph_argument_name_type_t nameType) const;

    /**
     * @brief Queries a single name of a graph argument through "pfnGetArgumentNames" (graph ext 1.20+).
     * @param fieldName Only used for reporting a malformed driver answer.
     */
    std::string getArgumentName(ze_graph_handle_t graphHandle,
                                uint32_t indexUsedByDriver,
                                ze_graph_argument_name_type_t nameType,
                                const char* fieldName) const;

    /**
     * @brief Queries the IR metadata of a graph argument.
     * @return `std::nullopt` for the arguments added by the compiler, which have no counterpart in the IR.
     */
    std::optional<ze_graph_argument_metadata_t> getArgumentMetadata(ze_graph_handle_t graphHandle,
                                                                    uint32_t indexUsedByDriver,
                                                                    std::string_view name) const;

    void initializeGraphThroughCommandList(ze_graph_handle_t graphHandle) const;

    bool canCpuVaBeImported(const void* data, size_t size) const;

    std::shared_ptr<ZeroInitStructsHolder> _zeroInitStruct;
    uint32_t _graphExtVersion;
    bool _isCompilerOptionQuerySupported;

    Logger _logger;
};

/**
 * @brief Extracts the I/O metadata from Level Zero specific structures and converts them into OpenVINO specific
 * ones.
 *
 * @param arg The main Level Zero structure from which most metadata will be extracted.
 * @param metadata The secondary Level Zero structure from which metadata will be extracted. More specifically, the
 * argument is used for populating "shapeFromIRModel". Not providing this argument will lead to an empty value for
 * the referenced attribute.
 * @returns A descriptor object containing the metadata converted in OpenVINO specific structures.
 */
IODescriptor createIODescriptorFromLevelZero(uint32_t indexUsedByDriver,
                                             const ze_graph_argument_properties_3_t& arg,
                                             const std::optional<ze_graph_argument_metadata_t>& metadata);

/**
 * @brief Same as the overload above, but for graph ext 1.20+ where the names are not part of the properties structure.
 *
 * @param arg The Level Zero structure from which the precision, the shape and the strides support are extracted.
 * @param name The name of the argument, as returned for "ZE_GRAPH_ARGUMENT_NAME".
 * @param debugFriendlyName The name returned for "ZE_GRAPH_ARGUMENT_DEBUG_FRIENDLY_NAME".
 * @param associatedTensorNames The names returned for "ZE_GRAPH_ARGUMENT_ASSOCIATED_TENSOR_NAMES", already split.
 * @param metadata The IR metadata of the argument. Unlike in the overload above it is required, as "arg" does not
 * report its rank, which is taken from here.
 */
IODescriptor createIODescriptorFromLevelZero(uint32_t indexUsedByDriver,
                                             const ze_graph_argument_properties_4_t& arg,
                                             std::string name,
                                             std::string debugFriendlyName,
                                             std::unordered_set<std::string> associatedTensorNames,
                                             const ze_graph_argument_metadata_t& metadata);

/**
 * @brief Splits the answer of "ZE_GRAPH_ARGUMENT_ASSOCIATED_TENSOR_NAMES" into a set of names.
 * @details The header documents a comma separated list, but the NUL character is a separator too, as drivers return
 * the names in NUL-padded fixed-size slots. Empty entries are dropped. A name holding a separator cannot be told apart
 * from several names.
 */
std::unordered_set<std::string> parseTensorNames(std::string_view names);

// Parse the result string of query from format <name_0><name_1><name_2> to unordered_set of string
std::unordered_set<std::string> parseQueryResult(std::vector<char>& data);

}  // namespace intel_npu
