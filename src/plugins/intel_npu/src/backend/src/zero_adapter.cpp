// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_adapter.hpp"

#include <regex>
#include <string_view>

#include "intel_npu/config/runtime.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
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

template <typename TableExtension>
ZeroAdapter<TableExtension>::ZeroAdapter(const std::shared_ptr<ZeroInitStructsHolder>& initStructs)
    : _initStructs(initStructs),
      _logger("ZeroAdapter", Logger::global().level()) {
    ze_device_properties_t deviceProperties = {};
    deviceProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_initStructs->getDevice(), &deviceProperties));
    _groupOrdinal = zeroUtils::findGroupOrdinal(_initStructs->getDevice(), deviceProperties);
}

template <typename TableExtension>
ZeroAdapter<TableExtension>::~ZeroAdapter() {
    _logger.debug("ZeroAdapter obj destroyed");
}

template <typename TableExtension>
_ze_result_t ZeroAdapter<TableExtension>::release(ze_graph_handle_t graphHandle) {
    _logger.debug("release - pfnDestroy graphHandle");
    auto result = _initStructs->getGraphDdiTable().pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("failed to release graph handle. L0 pfnDestroy result: %s, code %#X",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result));
    }

    return result;
}

template <typename TableExtension>
template <typename T, std::enable_if_t<UseCopyForNativeBinary(T), bool>>
void ZeroAdapter<TableExtension>::getNativeBinary(ze_graph_handle_t graphHandle,
                                                  std::vector<uint8_t>& blob,
                                                  const uint8_t*& blobPtr,
                                                  size_t& blobSize) const {
    // Get blob size first
    auto result = _initStructs->getGraphDdiTable().pfnGetNativeBinary(graphHandle, &blobSize, nullptr);
    blob.resize(blobSize);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob size, Failed to compile network.",
                                    result,
                                    _initStructs->getGraphDdiTable());

    // Get blob data
    result = _initStructs->getGraphDdiTable().pfnGetNativeBinary(graphHandle, &blobSize, blob.data());
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob data, Failed to compile network.",
                                    result,
                                    _initStructs->getGraphDdiTable());

    blobPtr = blob.data();
}

template <typename TableExtension>
template <typename T, std::enable_if_t<!UseCopyForNativeBinary(T), bool>>
void ZeroAdapter<TableExtension>::getNativeBinary(ze_graph_handle_t graphHandle,
                                                  std::vector<uint8_t>& /* unusedBlob */,
                                                  const uint8_t*& blobPtr,
                                                  size_t& blobSize) const {
    // Get blob ptr and size
    auto result = _initStructs->getGraphDdiTable().pfnGetNativeBinary2(graphHandle, &blobSize, &blobPtr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob size, Failed to compile network.",
                                    result,
                                    _initStructs->getGraphDdiTable());
}

template <typename TableExtension>
CompiledNetwork ZeroAdapter<TableExtension>::getCompiledNetwork(ze_graph_handle_t graphHandle) {
    if (graphHandle == nullptr) {
        OPENVINO_THROW("Graph handle is null");
    }

    _logger.info("ZeroAdapter getCompiledNetwork get blob from graphHandle");

    const uint8_t* blobPtr = nullptr;
    size_t blobSize = -1;
    std::vector<uint8_t> blob;

    getNativeBinary(graphHandle, blob, blobPtr, blobSize);

    _logger.info("ZeroAdapter getCompiledNetwork returning blob");
    return CompiledNetwork(blobPtr, blobSize, std::move(blob));
}

template <typename TableExtension>
void ZeroAdapter<TableExtension>::setArgumentValue(ze_graph_handle_t graphHandle,
                                                   uint32_t argi,
                                                   const void* argv) const {
    auto result = _initStructs->getGraphDdiTable().pfnSetArgumentValue(graphHandle, argi, argv);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("zeGraphSetArgumentValue", result, _initStructs->getGraphDdiTable());
}

template <typename TableExtension>
void ZeroAdapter<TableExtension>::graphInitialie(ze_graph_handle_t graphHandle, const Config& config) const {
    if (_initStructs->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8) {
        initialize_graph_through_command_list(graphHandle, config);
    } else {
        ze_graph_properties_2_t properties = {};
        properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        _initStructs->getGraphDdiTable().pfnGetProperties2(graphHandle, &properties);

        if (properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
            _initStructs->getGraphDdiTable().pfnGraphInitialize(graphHandle);
        }

        if (properties.initStageRequired & ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) {
            initialize_graph_through_command_list(graphHandle, config);
        }
    }
}

template <typename TableExtension>
void ZeroAdapter<TableExtension>::initialize_graph_through_command_list(ze_graph_handle_t graphHandle,
                                                                        const Config& config) const {
    _logger.debug("ZeroExecutor::ZeroExecutor init start - create graph_command_list");
    CommandList graph_command_list(_initStructs->getDevice(),
                                   _initStructs->getContext(),
                                   _initStructs->getGraphDdiTable(),
                                   _groupOrdinal);
    _logger.debug("ZeroExecutor::ZeroExecutor - create graph_command_queue");
    CommandQueue graph_command_queue(_initStructs->getDevice(),
                                     _initStructs->getContext(),
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                     _initStructs->getCommandQueueDdiTable(),
                                     false,
                                     _groupOrdinal);
    _logger.debug("ZeroExecutor::ZeroExecutor - create fence");
    Fence fence(graph_command_queue);

    _logger.debug("ZeroExecutor::ZeroExecutor - performing appendGraphInitialize");
    graph_command_list.appendGraphInitialize(graphHandle);
    _logger.debug("ZeroExecutor::ZeroExecutor - closing graph command list");
    graph_command_list.close();

    _logger.debug("ZeroExecutor::ZeroExecutor - performing executeCommandList");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    _logger.debug("ZeroExecutor::ZeroExecutor - performing hostSynchronize");
    fence.hostSynchronize();
    _logger.debug("ZeroExecutor::ZeroExecutor - hostSynchronize completed");
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
template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportQuery(T), bool>>
std::unordered_set<std::string> ZeroAdapter<TableExtension>::queryImpl(std::pair<size_t, std::shared_ptr<uint8_t>>,
                                                                       const std::string&) const {
    _logger.info("queryImpl - Driver version is less than 1.3, queryNetwork is unsupported.");
    return std::unordered_set<std::string>();
}

// For ext version == 1.3 && == 1.4
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
ze_result_t ZeroAdapter<TableExtension>::seriazlideIRModelAndQueryNetworkCreateV1(
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
    ze_result_t result = _initStructs->getGraphDdiTable().pfnQueryNetworkCreate(_initStructs->getContext(),
                                                                                _initStructs->getDevice(),
                                                                                &desc,
                                                                                &hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("seriazlideIRModelAndQueryNetworkCreateV1",
                                    result,
                                    _initStructs->getGraphDdiTable());

    return result;
}

// For ext version == 1.3 && == 1.4, query is supported, calling querynetwork api in _initStructs->getGraphDdiTable()
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
std::unordered_set<std::string> ZeroAdapter<TableExtension>::queryImpl(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags) const {
    _logger.info("queryImpl - Calling queryNetwork of 1.3 version.");

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    auto result = seriazlideIRModelAndQueryNetworkCreateV1(std::move(serializedIR), buildFlags, hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

// For ext version >= 1.5
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
ze_result_t ZeroAdapter<TableExtension>::seriazlideIRModelAndQueryNetworkCreateV2(
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
    _logger.debug("seriazlideIRModelAndQueryNetworkCreateV2 - performing pfnQueryNetworkCreate2");
    ze_result_t result = _initStructs->getGraphDdiTable().pfnQueryNetworkCreate2(_initStructs->getContext(),
                                                                                 _initStructs->getDevice(),
                                                                                 &desc,
                                                                                 &hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("seriazlideIRModelAndQueryNetworkCreateV2",
                                    result,
                                    _initStructs->getGraphDdiTable());

    return result;
}

// For ext version >= 1.5
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
std::unordered_set<std::string> ZeroAdapter<TableExtension>::queryImpl(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags) const {
    _logger.debug("queryImpl - Calling queryNetwork of 1.5 version.");

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    auto result = seriazlideIRModelAndQueryNetworkCreateV2(std::move(serializedIR), buildFlags, hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportQuery(T), bool>>
std::unordered_set<std::string> ZeroAdapter<TableExtension>::getQueryResultFromSupportedLayers(
    ze_result_t result,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    // Get the size of query result
    size_t size = 0;
    result = _initStructs->getGraphDdiTable().pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkGetSupportedLayers get size of query result",
                                    result,
                                    _initStructs->getGraphDdiTable());

    // Get the result data of query
    std::vector<char> supportedLayers(size);
    result = _initStructs->getGraphDdiTable().pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork,
                                                                                &size,
                                                                                supportedLayers.data());
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkGetSupportedLayers get result data of query",
                                    result,
                                    _initStructs->getGraphDdiTable());

    result = _initStructs->getGraphDdiTable().pfnQueryNetworkDestroy(hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkDestroy", result, _initStructs->getGraphDdiTable());

    return parseQueryResult(supportedLayers);
}

template <typename TableExtension>
std::unordered_set<std::string> ZeroAdapter<TableExtension>::queryResultFromSupportedLayers(
    std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
    const std::string& buildFlags) const {
    return queryImpl(std::move(serializedIR), buildFlags);
}

// For ext version <1.5, calling pfnCreate api in _initStructs->getGraphDdiTable()
template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportGraph2(T), bool>>
void ZeroAdapter<TableExtension>::createGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
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
    auto result =
        _initStructs->getGraphDdiTable().pfnCreate(_initStructs->getContext(), _initStructs->getDevice(), &desc, graph);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate", result, _initStructs->getGraphDdiTable());
}

// For ext version >= 1.5, calling pfnCreate2 api in _initStructs->getGraphDdiTable()
template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportGraph2(T), bool>>
void ZeroAdapter<TableExtension>::createGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
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
    auto result = _initStructs->getGraphDdiTable().pfnCreate2(_initStructs->getContext(),
                                                              _initStructs->getDevice(),
                                                              &desc,
                                                              graph);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate2", result, _initStructs->getGraphDdiTable());
}

template <typename TableExtension>
ze_graph_handle_t ZeroAdapter<TableExtension>::getGraphHandle(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                              const std::string& buildFlags,
                                                              const uint32_t& flags) const {
    ze_graph_handle_t graphHandle;

    _logger.info("compileIR Using extension version: %s", typeid(TableExtension).name());
    createGraph(std::move(serializedIR), buildFlags, flags, &graphHandle);

    return graphHandle;
}

template <typename TableExtension>
ze_graph_handle_t ZeroAdapter<TableExtension>::getGraphHandle(const std::vector<uint8_t>& network) const {
    ze_graph_handle_t graphHandle;

    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NATIVE,
                            network.size(),
                            network.data(),
                            nullptr};

    auto result = _initStructs->getGraphDdiTable().pfnCreate(_initStructs->getContext(),
                                                             _initStructs->getDevice(),
                                                             &desc,
                                                             &graphHandle);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate", result, _initStructs->getGraphDdiTable());

    return graphHandle;
}

/**
 * @brief Extracts the I/O metadata from Level Zero specific structures and converts them into OpenVINO specific ones.
 *
 * @param arg The main Level Zero structure from which most metadata will be extracted.
 * @param metadata The secondary Level Zero structure from which metadata will be extracted. More specifically, the
 * argument is used for populating "shapeFromIRModel". Not providing this argument will lead to an empty value for the
 * referenced attribute.
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

template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportArgumentMetadata(T), bool>>
void ZeroAdapter<TableExtension>::getMetadata(ze_graph_handle_t graphHandle,
                                              uint32_t index,
                                              std::vector<IODescriptor>& inputs,
                                              std::vector<IODescriptor>& outputs) const {
    ze_graph_argument_properties_3_t arg;
    auto result = _initStructs->getGraphDdiTable().pfnGetArgumentProperties3(graphHandle, index, &arg);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _initStructs->getGraphDdiTable());

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

template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportArgumentMetadata(T), bool>>
void ZeroAdapter<TableExtension>::getMetadata(ze_graph_handle_t graphHandle,
                                              uint32_t index,
                                              std::vector<IODescriptor>& inputs,
                                              std::vector<IODescriptor>& outputs) const {
    ze_graph_argument_properties_3_t arg;
    auto result = _initStructs->getGraphDdiTable().pfnGetArgumentProperties3(graphHandle, index, &arg);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _initStructs->getGraphDdiTable());

    std::optional<ze_graph_argument_metadata_t> optionalMetadata = std::nullopt;

    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name) && !isShapeTensorName(arg.name)) {
        ze_graph_argument_metadata_t metadata;
        result = _initStructs->getGraphDdiTable().pfnGraphGetArgumentMetadata(graphHandle, index, &metadata);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGraphGetArgumentMetadata", result, _initStructs->getGraphDdiTable());

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

template <typename TableExtension>
NetworkMetadata ZeroAdapter<TableExtension>::getNetworkMeta(ze_graph_handle_t graphHandle) const {
    ze_graph_properties_t graphProperties{};

    auto result = _initStructs->getGraphDdiTable().pfnGetProperties(graphHandle, &graphProperties);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _initStructs->getGraphDdiTable());

    NetworkMetadata meta;

    for (uint32_t index = 0; index < graphProperties.numGraphArgs; ++index) {
        getMetadata(graphHandle, index, meta.inputs, meta.outputs);
    }
    // TODO: support this information in CiD [track: E#33479]
    meta.numStreams = 1;
    meta.bindRelatedDescriptors();

    return meta;
}

template <typename TableExtension>
std::tuple<std::vector<ArgumentDescriptor>, std::vector<ArgumentDescriptor>> ZeroAdapter<TableExtension>::getIODesc(
    ze_graph_handle_t graphHandle) const {
    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    auto result = _initStructs->getGraphDdiTable().pfnGetProperties(graphHandle, &props);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _initStructs->getGraphDdiTable());

    std::vector<ArgumentDescriptor> input_descriptors;
    std::vector<ArgumentDescriptor> output_descriptors;

    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        auto result = _initStructs->getGraphDdiTable().pfnGetArgumentProperties3(graphHandle, index, &arg3);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _initStructs->getGraphDdiTable());

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            input_descriptors.push_back(ArgumentDescriptor{arg3, index});
        } else {
            output_descriptors.push_back(ArgumentDescriptor{arg3, index});
        }
    }

    return std::make_tuple(input_descriptors, output_descriptors);
}

template <typename TableExtension>
std::shared_ptr<CommandQueue> ZeroAdapter<TableExtension>::crateCommandQueue(const Config& config) const {
    if (config.has<TURBO>()) {
        bool turbo = config.get<TURBO>();
        return std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                              _initStructs->getContext(),
                                              zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                              _initStructs->getCommandQueueDdiTable(),
                                              turbo,
                                              _groupOrdinal);
    }

    return std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                          _initStructs->getContext(),
                                          zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                          _initStructs->getCommandQueueDdiTable(),
                                          false,
                                          _groupOrdinal);
}

template <typename TableExtension>
ze_device_graph_properties_t ZeroAdapter<TableExtension>::getDeviceGraphProperties() const {
    ze_device_graph_properties_t deviceGraphProperties = {};
    deviceGraphProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
    auto result =
        _initStructs->getGraphDdiTable().pfnDeviceGetGraphProperties(_initStructs->getDevice(), &deviceGraphProperties);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnDeviceGetGraphProperties", result, _initStructs->getGraphDdiTable());

    return deviceGraphProperties;
}

template class ZeroAdapter<ze_graph_dditable_ext_1_2_t>;
template class ZeroAdapter<ze_graph_dditable_ext_1_3_t>;
template class ZeroAdapter<ze_graph_dditable_ext_1_4_t>;
template class ZeroAdapter<ze_graph_dditable_ext_1_5_t>;
template class ZeroAdapter<ze_graph_dditable_ext_1_6_t>;
template class ZeroAdapter<ze_graph_dditable_ext_1_7_t>;
template class ZeroAdapter<ze_graph_dditable_ext_1_8_t>;

}  // namespace intel_npu
