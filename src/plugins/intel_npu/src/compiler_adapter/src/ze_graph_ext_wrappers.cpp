// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"

#define NotSupportQuery(T) (T <= ZE_GRAPH_EXT_VERSION_1_2)

// ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
// pfnQueryNetworkGetSupportedLayers)
#define SupportAPIGraphQueryNetworkV1(T) (T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4)

// ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
#define SupportAPIGraphQueryNetworkV2(T) ((!NotSupportQuery(T) && !SupportAPIGraphQueryNetworkV1(T)))

// For ext version >= 1.5, pfnCreate2 api is avaible
#define NotSupportGraph2(T) (T < ZE_GRAPH_EXT_VERSION_1_5)

// A bug inside the driver makes the "pfnGraphGetArgumentMetadata" call not safe for use prior to
// "ze_graph_dditable_ext_1_6_t".
// See: E#117498
#define NotSupportArgumentMetadata(T) (T < ZE_GRAPH_EXT_VERSION_1_6)

#define UseCopyForNativeBinary(T) (T < ZE_GRAPH_EXT_VERSION_1_7)

namespace {

ov::element::Type_t toOVElementType(const ze_graph_argument_precision_t zeElementType) {
    switch (zeElementType) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN:
        return ov::element::Type_t::boolean;
    case ZE_GRAPH_ARGUMENT_PRECISION_NF4:
        return ov::element::Type_t::nf4;
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
        return ov::element::Type_t::dynamic;
    }
}

}  // namespace

namespace intel_npu {

ZeGraphExtWrappers::ZeGraphExtWrappers(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _graphExtVersion(zeroInitStruct->getGraphDdiTable().version()),
      _logger("ZeGraphExtWrappers", Logger::global().level()) {
    _logger.info("Graph ext version used by zero wrapper: %d.%d",
                 ZE_MAJOR_VERSION(_graphExtVersion),
                 ZE_MINOR_VERSION(_graphExtVersion));
    _logger.debug("capabilities:");
    _logger.debug("-SupportQuery: %d", !NotSupportQuery(_graphExtVersion));
    _logger.debug("-SupportAPIGraphQueryNetworkV1: %d", SupportAPIGraphQueryNetworkV1(_graphExtVersion));
    _logger.debug("-SupportAPIGraphQueryNetworkV2 :%d", SupportAPIGraphQueryNetworkV2(_graphExtVersion));
    _logger.debug("-SupportpfnCreate2 :%d", !NotSupportGraph2(_graphExtVersion));
    _logger.debug("-SupportArgumentMetadata :%d", !NotSupportArgumentMetadata(_graphExtVersion));
    _logger.debug("-UseCopyForNativeBinary :%d", UseCopyForNativeBinary(_graphExtVersion));
}

ZeGraphExtWrappers::~ZeGraphExtWrappers() {
    _logger.debug("Obj destroyed");
}

_ze_result_t ZeGraphExtWrappers::destroyGraph(ze_graph_handle_t graphHandle) {
    _logger.debug("destroyGraph - perfrom pfnDestroy");
    auto result = _zeroInitStruct->getGraphDdiTable().pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("failed to destroy graph handle. L0 pfnDestroy result: %s, code %#X",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result));
    }

    return result;
}

void ZeGraphExtWrappers::getGraphBinary(ze_graph_handle_t graphHandle,
                                        std::vector<uint8_t>& blob,
                                        const uint8_t*& blobPtr,
                                        size_t& blobSize) const {
    if (graphHandle == nullptr) {
        OPENVINO_THROW("Graph handle is null");
    }

    _logger.debug("getGraphBinary - get blob from graphHandle");

    if (UseCopyForNativeBinary(_graphExtVersion)) {
        // Get blob size first
        _logger.debug("getGraphBinary - perfrom pfnGetNativeBinary to get size");
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetNativeBinary(graphHandle, &blobSize, nullptr);
        blob.resize(blobSize);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob size, Failed to compile network.",
                                        result,
                                        _zeroInitStruct->getGraphDdiTable());

        // Get blob data
        _logger.debug("getGraphBinary - perfrom pfnGetNativeBinary to get data");
        result = _zeroInitStruct->getGraphDdiTable().pfnGetNativeBinary(graphHandle, &blobSize, blob.data());
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob data, Failed to compile network.",
                                        result,
                                        _zeroInitStruct->getGraphDdiTable());

        blobPtr = blob.data();
    } else {
        // Get blob ptr and size
        _logger.debug("getGraphBinary - perfrom pfnGetNativeBinary2 to get size and data");
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetNativeBinary2(graphHandle, &blobSize, &blobPtr);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetNativeBinary get blob size, Failed to compile network.",
                                        result,
                                        _zeroInitStruct->getGraphDdiTable());
    }
}

void ZeGraphExtWrappers::setGraphArgumentValue(ze_graph_handle_t graphHandle, uint32_t argi, const void* argv) const {
    _logger.debug("setGraphArgumentValue - perform pfnSetArgumentValue");
    auto result = _zeroInitStruct->getGraphDdiTable().pfnSetArgumentValue(graphHandle, argi, argv);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("zeGraphSetArgumentValue", result, _zeroInitStruct->getGraphDdiTable());
}

void ZeGraphExtWrappers::initializeGraph(ze_graph_handle_t graphHandle, uint32_t commandQueueGroupOrdinal) const {
    if (_zeroInitStruct->getGraphDdiTable().version() < ZE_GRAPH_EXT_VERSION_1_8) {
        _logger.debug("Use initialize_graph_through_command_list for ext version smaller than 1.8");
        initialize_graph_through_command_list(graphHandle, commandQueueGroupOrdinal);
    } else {
        _logger.debug("Initialize graph based on graph properties for ext version larger than 1.8");
        ze_graph_properties_2_t properties = {};
        properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        _logger.debug("initializeGraph - perfrom pfnGetProperties2");
        _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(graphHandle, &properties);

        if (properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
            _logger.debug("initializeGraph - perfrom pfnGraphInitialize");
            _zeroInitStruct->getGraphDdiTable().pfnGraphInitialize(graphHandle);
        }

        if (properties.initStageRequired & ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) {
            initialize_graph_through_command_list(graphHandle, commandQueueGroupOrdinal);
        }
    }
}

void ZeGraphExtWrappers::initialize_graph_through_command_list(ze_graph_handle_t graphHandle,
                                                               uint32_t commandQueueGroupOrdinal) const {
    _logger.debug("initialize_graph_through_command_list init start - create graph_command_list");
    CommandList graph_command_list(_zeroInitStruct, commandQueueGroupOrdinal);
    _logger.debug("initialize_graph_through_command_list - create graph_command_queue");
    std::shared_ptr<CommandQueue> graph_command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                                                       ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                                                                       commandQueueGroupOrdinal,
                                                                                       false);
    _logger.debug("initialize_graph_through_command_list - create fence");
    Fence fence(graph_command_queue);

    _logger.debug("initialize_graph_through_command_list - performing appendGraphInitialize");
    graph_command_list.appendGraphInitialize(graphHandle);
    _logger.debug("initialize_graph_through_command_list - closing graph command list");
    graph_command_list.close();

    _logger.debug("initialize_graph_through_command_list - performing executeCommandList");
    graph_command_queue->executeCommandList(graph_command_list, fence);
    _logger.debug("initialize_graph_through_command_list - performing hostSynchronize");
    fence.hostSynchronize();
    _logger.debug("initialize_graph_through_command_list - hostSynchronize completed");
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
            result.insert(std::move(temp));
            i++;
        } else {
            i++;
        }
    }
    return result;
}

std::unordered_set<std::string> ZeGraphExtWrappers::getQueryResultFromSupportedLayers(
    ze_result_t result,
    ze_graph_query_network_handle_t& hGraphQueryNetwork) const {
    if (NotSupportQuery(_graphExtVersion)) {
        OPENVINO_THROW("pfnQueryNetworkGetSupportedLayers not supported for ",
                       ZE_MAJOR_VERSION(_graphExtVersion),
                       ".",
                       ZE_MINOR_VERSION(_graphExtVersion));
    }
    // Get the size of query result
    _logger.debug("getQueryResultFromSupportLayers - perfrom pfnQueryNetworkGetSupportedLayers to get size");
    size_t size = 0;
    result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, nullptr);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkGetSupportedLayers get size of query result",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());

    // Get the result data of query
    _logger.debug("getQueryResultFromSupportLayers - perfrom pfnQueryNetworkGetSupportedLayers to get data");
    std::vector<char> supportedLayers(size);
    result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork,
                                                                                   &size,
                                                                                   supportedLayers.data());
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkGetSupportedLayers get result data of query",
                                    result,
                                    _zeroInitStruct->getGraphDdiTable());

    _logger.debug("getQueryResultFromSupportLayers - perfrom pfnQueryNetworkDestroy");
    result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkDestroy(hGraphQueryNetwork);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkDestroy", result, _zeroInitStruct->getGraphDdiTable());

    return parseQueryResult(supportedLayers);
}

std::unordered_set<std::string> ZeGraphExtWrappers::queryGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                               const std::string& buildFlags) const {
    // ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
    // ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
    // pfnQueryNetworkGetSupportedLayers)
    // For ext version < 1.3, query is not supported
    ze_result_t result = ZE_RESULT_SUCCESS;
    if (NotSupportQuery(_graphExtVersion)) {
        // For ext version < 1.3, query is unsupported, return empty result and add debug log here
        _logger.warning("queryGraph - Driver version is less than 1.3, queryNetwork is unsupported.");
        return std::unordered_set<std::string>();
    } else if (SupportAPIGraphQueryNetworkV1(_graphExtVersion)) {
        // For ext version == 1.3 && == 1.4, query is supported, calling querynetwork api in
        // _zeroInitStruct->getGraphDdiTable()
        ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

        // For ext version == 1.3 && == 1.4
        ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                                nullptr,
                                ZE_GRAPH_FORMAT_NGRAPH_LITE,
                                serializedIR.first,
                                serializedIR.second.get(),
                                buildFlags.c_str()};

        // Create querynetwork handle
        _logger.debug("For ext of 1.3 and 1.4 - perform pfnQueryNetworkCreate");
        result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkCreate(_zeroInitStruct->getContext(),
                                                                           _zeroInitStruct->getDevice(),
                                                                           &desc,
                                                                           &hGraphQueryNetwork);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkCreate", result, _zeroInitStruct->getGraphDdiTable());

        return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
    } else if (SupportAPIGraphQueryNetworkV2(_graphExtVersion)) {
        // For ext version >= 1.5
        ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

        // For ext version >= 1.5
        ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                                  nullptr,
                                  ZE_GRAPH_FORMAT_NGRAPH_LITE,
                                  serializedIR.first,
                                  serializedIR.second.get(),
                                  buildFlags.c_str(),
                                  ZE_GRAPH_FLAG_NONE};

        // Create querynetwork handle
        _logger.debug("For ext larger than 1.4 - perform pfnQueryNetworkCreate2");
        result = _zeroInitStruct->getGraphDdiTable().pfnQueryNetworkCreate2(_zeroInitStruct->getContext(),
                                                                            _zeroInitStruct->getDevice(),
                                                                            &desc,
                                                                            &hGraphQueryNetwork);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryNetworkCreate2", result, _zeroInitStruct->getGraphDdiTable());

        return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
    }
    _logger.warning("queryGraph - Driver version is %d.%d, queryNetwork is unsupported.",
                    ZE_MAJOR_VERSION(_graphExtVersion),
                    ZE_MINOR_VERSION(_graphExtVersion));
    return std::unordered_set<std::string>();
}

ze_graph_handle_t ZeGraphExtWrappers::getGraphHandle(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                     const std::string& buildFlags,
                                                     const uint32_t& flags) const {
    ze_graph_handle_t graphHandle;
    if (NotSupportGraph2(_graphExtVersion)) {
        // For ext version <1.5, calling pfnCreate api in _zeroInitStruct->getGraphDdiTable()
        ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                                nullptr,
                                ZE_GRAPH_FORMAT_NGRAPH_LITE,
                                serializedIR.first,
                                serializedIR.second.get(),
                                buildFlags.c_str()};

        _logger.debug("getGraphHandle - perform pfnCreate");
        // Create querynetwork handle
        auto result = _zeroInitStruct->getGraphDdiTable().pfnCreate(_zeroInitStruct->getContext(),
                                                                    _zeroInitStruct->getDevice(),
                                                                    &desc,
                                                                    &graphHandle);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate", result, _zeroInitStruct->getGraphDdiTable());
    } else {
        // For ext version >= 1.5, calling pfnCreate2 api in _zeroInitStruct->getGraphDdiTable()
        ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                                  nullptr,
                                  ZE_GRAPH_FORMAT_NGRAPH_LITE,
                                  serializedIR.first,
                                  serializedIR.second.get(),
                                  buildFlags.c_str(),
                                  flags};

        _logger.debug("getGraphHandle - perform pfnCreate2");
        // Create querynetwork handle
        auto result = _zeroInitStruct->getGraphDdiTable().pfnCreate2(_zeroInitStruct->getContext(),
                                                                     _zeroInitStruct->getDevice(),
                                                                     &desc,
                                                                     &graphHandle);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate2", result, _zeroInitStruct->getGraphDdiTable());
    }
    return graphHandle;
}

ze_graph_handle_t ZeGraphExtWrappers::getGraphHandle(const uint8_t& blobData, size_t blobSize) const {
    ze_graph_handle_t graphHandle;

    if (blobSize == 0) {
        OPENVINO_THROW("Empty blob");
    }

    ze_graph_desc_t desc =
        {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES, nullptr, ZE_GRAPH_FORMAT_NATIVE, blobSize, &blobData, nullptr};

    _logger.debug("getGraphHandle - perform pfnCreate");
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
    ov::Shape shapeFromCompiler;
    ov::PartialShape shapeFromIRModel;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        outputTensorNames.insert(arg.associated_tensor_names[id]);
    }
    for (uint32_t id = 0; id < arg.dims_count; id++) {
        shapeFromCompiler.push_back(arg.dims[id]);
    }
    if (metadata.has_value()) {
        const auto dynamicDim = std::numeric_limits<uint64_t>::max();
        shapeFromIRModel.reserve(metadata->shape_size);
        for (uint32_t id = 0; id < metadata->shape_size; id++) {
            if (metadata->shape[id] != dynamicDim) {
                shapeFromIRModel.push_back(metadata->shape[id]);
            } else {
                // lower bound is ignored, so we set it to 1 just to satisfy the Dimension constructor,
                // upper bound is set to the value from shapeFromCompiler as it is filled with upper bounds
                // in case of dynamic dimensions
                shapeFromIRModel.push_back(ov::Dimension(1, shapeFromCompiler[id]));
            }
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
            shapeFromCompiler,
            isStateInput,
            isStateOutput,
            isShapeTensor,
            std::nullopt,
            arg.debug_friendly_name,
            std::move(outputTensorNames),
            metadata.has_value() ? std::optional(shapeFromIRModel) : std::nullopt};
}

void ZeGraphExtWrappers::getMetadata(ze_graph_handle_t graphHandle,
                                     uint32_t index,
                                     std::vector<IODescriptor>& inputs,
                                     std::vector<IODescriptor>& outputs) const {
    if (NotSupportArgumentMetadata(_graphExtVersion)) {
        ze_graph_argument_properties_3_t arg;
        _logger.debug("getMetadata - perfrom pfnGetArgumentProperties3");
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
            OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ",
                           arg.type);
        }
        }
    } else {
        ze_graph_argument_properties_3_t arg;
        _logger.debug("getMetadata - perfrom pfnGetArgumentProperties3");
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(graphHandle, index, &arg);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

        std::optional<ze_graph_argument_metadata_t> optionalMetadata = std::nullopt;

        if (!isStateInputName(arg.name) && !isStateOutputName(arg.name) && !isShapeTensorName(arg.name)) {
            _logger.debug("getMetadata - perfrom pfnGetArgumentMetadata");
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
            OPENVINO_THROW("Invalid ze_graph_argument_type_t found in ze_graph_argument_properties_3_t object: ",
                           arg.type);
        }
        }
    }
}

NetworkMetadata ZeGraphExtWrappers::getNetworkMeta(ze_graph_handle_t graphHandle) const {
    ze_graph_properties_t graphProperties{};

    _logger.debug("getNetworkMeta - perfrom pfnGetProperties");
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

}  // namespace intel_npu
