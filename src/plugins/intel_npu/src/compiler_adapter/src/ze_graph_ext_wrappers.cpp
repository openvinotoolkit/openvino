// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_graph_ext_wrappers.hpp"

#include <regex>
#include <string_view>

#include "intel_npu/config/runtime.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/core/model.hpp"

namespace {

ov::element::Type_t toOVElementType(const ze_graph_argument_precision_t zeElementType) {
    switch (zeElementType) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN:
        return ov::element::Type_t::undefined;
    case ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN:
        return ov::element::Type_t::boolean;
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return ov::element::Type_t::bf16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return ov::element::Type_t::f16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return ov::element::Type_t::f32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        return ov::element::Type_t::f64;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
        return ov::element::Type_t::i4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return ov::element::Type_t::i8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return ov::element::Type_t::i16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return ov::element::Type_t::i32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
        return ov::element::Type_t::i64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return ov::element::Type_t::u1;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
        return ov::element::Type_t::u4;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return ov::element::Type_t::u8;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return ov::element::Type_t::u16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        return ov::element::Type_t::u32;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        return ov::element::Type_t::u64;
    default:
        return ov::element::Type_t::undefined;
    }
}

}  // namespace

namespace intel_npu {

template <ze_graph_ext_version_t TableExtension>
ZeGraphExtWrappers<TableExtension>::ZeGraphExtWrappers(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("ZeGraphExtWrappers", Logger::global().level()) {}

template <ze_graph_ext_version_t TableExtension>
ZeGraphExtWrappers<TableExtension>::~ZeGraphExtWrappers() {
    _logger.debug("ZeGraphExtWrappers obj destroyed");
}

template <ze_graph_ext_version_t TableExtension>
_ze_result_t ZeGraphExtWrappers<TableExtension>::destroyGraph(ze_graph_handle_t graphHandle) {
    _logger.debug("destroyGraph - pfnDestroy graphHandle");
    auto result = _zeroInitStruct->getGraphDdiTable().pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("failed to destroy graph handle. L0 pfnDestroy result: %s, code %#X",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result));
    }

    return result;
}

template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<UseCopyForNativeBinary(T), bool>>
void ZeGraphExtWrappers<TableExtension>::getNativeBinary(ze_graph_handle_t graphHandle,
                                                         std::vector<uint8_t>& blob,
                                                         const uint8_t*& blobPtr,
                                                         size_t& blobSize) const {
    // Get blob size first
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetNativeBinary(graphHandle, &blobSize, nullptr);
    blob.resize(blobSize);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob size, Failed to compile network.",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());

    // Get blob data
    result = _zeroInitStruct->getGraphDdiTable().pfnGetNativeBinary(graphHandle, &blobSize, blob.data());
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob data, Failed to compile network.",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());

    blobPtr = blob.data();
}

template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<!UseCopyForNativeBinary(T), bool>>
void ZeGraphExtWrappers<TableExtension>::getNativeBinary(ze_graph_handle_t graphHandle,
                                                         std::vector<uint8_t>& /* unusedBlob */,
                                                         const uint8_t*& blobPtr,
                                                         size_t& blobSize) const {
    // Get blob ptr and size
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetNativeBinary2(graphHandle, &blobSize, &blobPtr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob size, Failed to compile network.",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());
}

template <ze_graph_ext_version_t TableExtension>
void ZeGraphExtWrappers<TableExtension>::getGraphBinary(ze_graph_handle_t graphHandle,
                                                        std::vector<uint8_t>& blob,
                                                        const uint8_t*& blobPtr,
                                                        size_t& blobSize) const {
    if (graphHandle == nullptr) {
        OPENVINO_THROW("Graph handle is null");
    }

    _logger.info("ZeGraphExtWrappers getGraphBinary get blob from graphHandle");

    getNativeBinary(graphHandle, blob, blobPtr, blobSize);
}

template <ze_graph_ext_version_t TableExtension>
void ZeGraphExtWrappers<TableExtension>::setGraphArgumentValue(ze_graph_handle_t graphHandle,
                                                               uint32_t argi,
                                                               const void* argv) const {
    auto result = _zeroInitStruct->getGraphDdiTable().pfnSetArgumentValue(graphHandle, argi, argv);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("zeGraphSetArgumentValue", result, _zeroInitStruct->getGraphDdiTable());
}

template <ze_graph_ext_version_t TableExtension>
void ZeGraphExtWrappers<TableExtension>::initializeGraph(ze_graph_handle_t graphHandle, const Config& config) const {
    if (_zeroInitStruct->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8) {
        initialize_graph_through_command_list(graphHandle, config);
    } else {
        ze_graph_properties_2_t properties = {};
        properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(graphHandle, &properties);

        if (properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
            _zeroInitStruct->getGraphDdiTable().pfnGraphInitialize(graphHandle);
        }

        if (properties.initStageRequired & ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) {
            initialize_graph_through_command_list(graphHandle, config);
        }
    }
}

template <ze_graph_ext_version_t TableExtension>
void ZeGraphExtWrappers<TableExtension>::initialize_graph_through_command_list(ze_graph_handle_t graphHandle,
                                                                               const Config& config) const {
    ze_device_properties_t deviceProperties = {};
    deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_zeroInitStruct->getDevice(), &deviceProperties));
    auto groupOrdinal = zeroUtils::findGroupOrdinal(_zeroInitStruct->getDevice(), deviceProperties);

    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list init start - create graph_command_list");
    CommandList graph_command_list(_zeroInitStruct->getDevice(),
                                   _zeroInitStruct->getContext(),
                                   _zeroInitStruct->getGraphDdiTable(),
                                   groupOrdinal);
    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - create graph_command_queue");
    CommandQueue graph_command_queue(_zeroInitStruct->getDevice(),
                                     _zeroInitStruct->getContext(),
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                     _zeroInitStruct->getCommandQueueDdiTable(),
                                     false,
                                     groupOrdinal);
    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - create fence");
    Fence fence(graph_command_queue);

    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - performing appendGraphInitialize");
    graph_command_list.appendGraphInitialize(graphHandle);
    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - closing graph command list");
    graph_command_list.close();

    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - performing executeCommandList");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - performing hostSynchronize");
    fence.hostSynchronize();
    _logger.debug("ZeGraphExtWrappers::initialize_graph_through_command_list - hostSynchronize completed");
}

// Parse the result string of query from foramt <name_0><name_1><name_2> to unordered_set of string
static std::unordered_set<std::string> parseQueryResult(std::vector<char>& data) {
    std::string dataString(data.begin(), data.end());
    std::unordered_set<std::string> result;
    size_t i = 0, start = 0;
    while (i < dataString.length()) {
        if (dataString[i] == '<') {
            start = ++i;
        } else if (dataString[i] == '>') {
            std::string temp(dataString.begin() + start, dataString.begin() + i);
            result.insert(temp);
            i++;
        } else {
            i++;
        }
    }
    return result;
}

// For ext version < 1.3, query is unsupported, return empty result and add debug log here
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<NotSupportQuery(T), bool>>
std::unordered_set<std::string> ZeGraphExtWrappers<TableExtension>::queryImpl(
    std::pair<size_t, std::shared_ptr<uint8_t>>,
    const std::string&) const {
    _logger.info("queryImpl - Driver version is less than 1.3, queryNetwork is unsupported.");
    return std::unordered_set<std::string>();
}

// For ext version == 1.3 && == 1.4
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
ze_result_t ZeGraphExtWrappers<TableExtension>::queryNetworkCreateV1(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            serializedIR.first,
                            serializedIR.second.get(),
                            buildFlags.c_str()};

    // Create querynetwork handle
    ze_result_t result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkCreate(_zeroInitStruct->getContext(),
                                                                                   _zeroInitStruct->getDevice(),
                                                                                   &desc,
                                                                                   &hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("queryNetworkCreateV1", result, _zeroInitStruct->getGraphDdiTable());

    return result;
}

// For ext version == 1.3 && == 1.4, query is supported, calling querynetwork api in _zeroInitStruct->getGraphDdiTable()
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
std::unordered_set<std::string> ZeGraphExtWrappers<TableExtension>::queryImpl(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags) const {
    _logger.info("queryImpl - Calling queryNetwork of 1.3 version.");

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    auto result = queryNetworkCreateV1(std::move(serializedIR), buildFlags, hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

// For ext version >= 1.5
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
ze_result_t ZeGraphExtWrappers<TableExtension>::queryNetworkCreateV2(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                              nullptr,
                              ZE_GRAPH_FORMAT_NGRAPH_LITE,
                              serializedIR.first,
                              serializedIR.second.get(),
                              buildFlags.c_str(),
                              ZE_GRAPH_FLAG_NONE};

    // Create querynetwork handle
    _logger.debug("queryNetworkCreateV2 - performing pfnQueryNetworkCreate2");
    ze_result_t result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkCreate2(_zeroInitStruct->getContext(),
                                                                                    _zeroInitStruct->getDevice(),
                                                                                    &desc,
                                                                                    &hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("queryNetworkCreateV2", result, _zeroInitStruct->getGraphDdiTable());

    return result;
}

// For ext version >= 1.5
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
std::unordered_set<std::string> ZeGraphExtWrappers<TableExtension>::queryImpl(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags) const {
    _logger.debug("queryImpl - Calling queryNetwork of 1.5 version.");

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    auto result = queryNetworkCreateV2(std::move(serializedIR), buildFlags, hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<!NotSupportQuery(T), bool>>
std::unordered_set<std::string> ZeGraphExtWrappers<TableExtension>::getQueryResultFromSupportedLayers(
    ze_result_t result,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    // Get the size of query result
    size_t size = 0;
    result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkGetSupportedLayers get size of query result",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());

    // Get the result data of query
    std::vector<char> supportedLayers(size);
    result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork,
                                                                                   &size,
                                                                                   supportedLayers.data());
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkGetSupportedLayers get result data of query",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());

    result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkDestroy(hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkDestroy", result, _zeroInitStruct->getGraphDdiTable());

    return parseQueryResult(supportedLayers);
}

template <ze_graph_ext_version_t TableExtension>
std::unordered_set<std::string> ZeGraphExtWrappers<TableExtension>::queryGraph(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags) const {
    return queryImpl(std::move(serializedIR), buildFlags);
}

// For ext version <1.5, calling pfnCreate api in _zeroInitStruct->getGraphDdiTable()
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<NotSupportGraph2(T), bool>>
void ZeGraphExtWrappers<TableExtension>::createGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                     const std::string& buildFlags,
                                                     const uint32_t& /*flags*/,
                                                     ze_graph_handle_t* graph) const {
    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            serializedIR.first,
                            serializedIR.second.get(),
                            buildFlags.c_str()};

    _logger.debug("createGraph - performing pfnCreate");
    // Create querynetwork handle
    auto result = _zeroInitStruct->getGraphDdiTable().pfnCreate(_zeroInitStruct->getContext(),
                                                                _zeroInitStruct->getDevice(),
                                                                &desc,
                                                                graph);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate", result, _zeroInitStruct->getGraphDdiTable());
}

// For ext version >= 1.5, calling pfnCreate2 api in _zeroInitStruct->getGraphDdiTable()
template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<!NotSupportGraph2(T), bool>>
void ZeGraphExtWrappers<TableExtension>::createGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                     const std::string& buildFlags,
                                                     const uint32_t& flags,
                                                     ze_graph_handle_t* graph) const {
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                              nullptr,
                              ZE_GRAPH_FORMAT_NGRAPH_LITE,
                              serializedIR.first,
                              serializedIR.second.get(),
                              buildFlags.c_str(),
                              flags};

    _logger.debug("createGraph - performing pfnCreate2");
    // Create querynetwork handle
    auto result = _zeroInitStruct->getGraphDdiTable().pfnCreate2(_zeroInitStruct->getContext(),
                                                                 _zeroInitStruct->getDevice(),
                                                                 &desc,
                                                                 graph);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate2", result, _zeroInitStruct->getGraphDdiTable());
}

template <ze_graph_ext_version_t TableExtension>
ze_graph_handle_t ZeGraphExtWrappers<TableExtension>::getGraphHandle(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags,
    const uint32_t& flags) const {
    ze_graph_handle_t graphHandle;

    _logger.info("compileIR Using extension version: %s", typeid(TableExtension).name());
    createGraph(std::move(serializedIR), buildFlags, flags, &graphHandle);

    return graphHandle;
}

template <ze_graph_ext_version_t TableExtension>
ze_graph_handle_t ZeGraphExtWrappers<TableExtension>::getGraphHandle(const std::vector<uint8_t>& network) const {
    ze_graph_handle_t graphHandle;

    if (network.empty()) {
        OPENVINO_THROW("Empty blob");
    }

    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NATIVE,
                            network.size(),
                            network.data(),
                            nullptr};

    auto result = _zeroInitStruct->getGraphDdiTable().pfnCreate(_zeroInitStruct->getContext(),
                                                                _zeroInitStruct->getDevice(),
                                                                &desc,
                                                                &graphHandle);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate", result, _zeroInitStruct->getGraphDdiTable());

    return graphHandle;
}

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
static IODescriptor getIODescriptor(const ze_graph_argument_properties_3_t& arg,
                                    const std::optional<ze_graph_argument_metadata_t>& metadata) {
    ov::element::Type_t precision = toOVElementType(arg.devicePrecision);
    ov::Shape shapeFromCompiler, shapeFromIRModel;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        outputTensorNames.insert(arg.associated_tensor_names[id]);
    }
    for (uint32_t id = 0; id < arg.dims_count; id++) {
        shapeFromCompiler.push_back(arg.dims[id]);
    }
    if (metadata.has_value()) {
        for (uint32_t id = 0; id < metadata->shape_size; id++) {
            shapeFromIRModel.push_back(metadata->shape[id]);
        }
    }

    // Flags will be used instead of indices for informing the type of the current entry
    std::string nameFromCompiler = arg.name;
    bool isStateInput = false;
    bool isStateOutput = false;
    bool isShapeTensor = false;
    if (isStateInputName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(READVALUE_PREFIX.length());
        isStateInput = true;
    } else if (isStateOutputName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(ASSIGN_PREFIX.length());
        isStateOutput = true;
    } else if (isShapeTensorName(nameFromCompiler)) {
        nameFromCompiler = nameFromCompiler.substr(SHAPE_TENSOR_PREFIX.length());
        isShapeTensor = true;
    }

    return {std::move(nameFromCompiler),
            precision,
            std::move(shapeFromCompiler),
            isStateInput,
            isStateOutput,
            isShapeTensor,
            std::nullopt,
            arg.debug_friendly_name,
            std::move(outputTensorNames),
            metadata.has_value() ? std::optional(shapeFromIRModel) : std::nullopt};
}

template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<NotSupportArgumentMetadata(T), bool>>
void ZeGraphExtWrappers<TableExtension>::getMetadata(ze_graph_handle_t graphHandle,
                                                     uint32_t index,
                                                     std::vector<IODescriptor>& inputs,
                                                     std::vector<IODescriptor>& outputs) const {
    ze_graph_argument_properties_3_t arg;
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(graphHandle, index, &arg);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

    switch (arg.type) {
    case ZE_GRAPH_ARGUMENT_TYPE_INPUT: {
        inputs.push_back(getIODescriptor(arg, std::nullopt));
    } break;
    case ZE_GRAPH_ARGUMENT_TYPE_OUTPUT: {
        outputs.push_back(getIODescriptor(arg, std::nullopt));
    } break;
    default: {
        OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ", arg.type);
    }
    }
}

template <ze_graph_ext_version_t TableExtension>
template <ze_graph_ext_version_t T, std::enable_if_t<!NotSupportArgumentMetadata(T), bool>>
void ZeGraphExtWrappers<TableExtension>::getMetadata(ze_graph_handle_t graphHandle,
                                                     uint32_t index,
                                                     std::vector<IODescriptor>& inputs,
                                                     std::vector<IODescriptor>& outputs) const {
    ze_graph_argument_properties_3_t arg;
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(graphHandle, index, &arg);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

    std::optional<ze_graph_argument_metadata_t> optionalMetadata = std::nullopt;

    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name) && !isShapeTensorName(arg.name)) {
        ze_graph_argument_metadata_t metadata;
        result = _zeroInitStruct->getGraphDdiTable().pfnGraphGetArgumentMetadata(graphHandle, index, &metadata);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGraphGetArgumentMetadata", result, _zeroInitStruct->getGraphDdiTable());

        optionalMetadata = std::optional(metadata);
    }

    switch (arg.type) {
    case ZE_GRAPH_ARGUMENT_TYPE_INPUT: {
        inputs.push_back(getIODescriptor(arg, optionalMetadata));
    } break;
    case ZE_GRAPH_ARGUMENT_TYPE_OUTPUT: {
        outputs.push_back(getIODescriptor(arg, optionalMetadata));
    } break;
    default: {
        OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ", arg.type);
    }
    }
}

template <ze_graph_ext_version_t TableExtension>
NetworkMetadata ZeGraphExtWrappers<TableExtension>::getNetworkMeta(ze_graph_handle_t graphHandle) const {
    ze_graph_properties_t graphProperties{};

    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties(graphHandle, &graphProperties);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

    NetworkMetadata meta;

    for (uint32_t index = 0; index < graphProperties.numGraphArgs; ++index) {
        getMetadata(graphHandle, index, meta.inputs, meta.outputs);
    }
    // TODO: support this information in CiD [track: E#33479]
    meta.numStreams = 1;
    meta.bindRelatedDescriptors();

    return meta;
}

template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_2>;
template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_3>;
template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_4>;
template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_5>;
template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_6>;
template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_7>;
template class ZeGraphExtWrappers<ZE_GRAPH_EXT_VERSION_1_8>;

}  // namespace intel_npu
