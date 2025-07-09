// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <fstream>
#include <regex>
#include <string_view>

#include "graph.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "ir_serializer.hpp"
#include "mem_usage.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "weightless_graph.hpp"

namespace {

constexpr std::string_view INPUTS_PRECISIONS_KEY = "--inputs_precisions";
constexpr std::string_view INPUTS_LAYOUTS_KEY = "--inputs_layouts";
constexpr std::string_view OUTPUTS_PRECISIONS_KEY = "--outputs_precisions";
constexpr std::string_view OUTPUTS_LAYOUTS_KEY = "--outputs_layouts";

// <option key>="<option value>"
constexpr std::string_view KEY_VALUE_SEPARATOR = "=";
constexpr std::string_view VALUE_DELIMITER = "\"";  // marks beginning and end of value

// Format inside "<option value>"
// <name1>:<value (precision / layout)> [<name2>:<value>]
constexpr std::string_view NAME_VALUE_SEPARATOR = ":";
constexpr std::string_view VALUES_SEPARATOR = " ";

// Constants indicating the order indices needed to be applied as to perform conversions between legacy layout values
const std::vector<size_t> NC_TO_CN_LAYOUT_DIMENSIONS_ORDER = {1, 0};
const std::vector<size_t> NCHW_TO_NHWC_LAYOUT_DIMENSIONS_ORDER = {0, 2, 3, 1};
const std::vector<size_t> NCDHW_TO_NDHWC_LAYOUT_DIMENSIONS_ORDER = {0, 2, 3, 4, 1};

/**
 * @brief A standard copy function concerning memory segments. Additional checks on the given arguments are performed
 * before copying.
 * @details This is meant as a replacement for the legacy "ie_memcpy" function coming from the OpenVINO API.
 */
void checkedMemcpy(void* destination, size_t destinationSize, const void* source, size_t numberOfBytes) {
    if (numberOfBytes == 0) {
        return;
    }

    OPENVINO_ASSERT(destination != nullptr, "Memcpy: received a null destination address");
    OPENVINO_ASSERT(source != nullptr, "Memcpy: received a null source address");
    OPENVINO_ASSERT(numberOfBytes <= destinationSize,
                    "Memcpy: the source buffer does not fit inside the destination one");
    OPENVINO_ASSERT(numberOfBytes <= (destination > source ? ((uintptr_t)destination - (uintptr_t)source)
                                                           : ((uintptr_t)source - (uintptr_t)destination)),
                    "Memcpy: the offset between the two buffers does not allow a safe execution of the operation");

    memcpy(destination, source, numberOfBytes);
}

/**
 * @brief For driver backward compatibility reasons, the given value shall be converted to a string corresponding to the
 * adequate legacy precision.
 */
std::string ovPrecisionToLegacyPrecisionString(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::f16:
        return "FP16";
    case ov::element::Type_t::f32:
        return "FP32";
    case ov::element::Type_t::f64:
        return "FP64";
    case ov::element::Type_t::bf16:
        return "BF16";
    case ov::element::Type_t::f8e4m3:
        return "FP8_E4M3";
    case ov::element::Type_t::f8e5m2:
        return "FP8_E5M2";
    case ov::element::Type_t::f8e8m0:
        return "FP8_E8M0";
    case ov::element::Type_t::nf4:
        return "NF4";
    case ov::element::Type_t::i4:
        return "I4";
    case ov::element::Type_t::i8:
        return "I8";
    case ov::element::Type_t::i16:
        return "I16";
    case ov::element::Type_t::i32:
        return "I32";
    case ov::element::Type_t::i64:
        return "I64";
    case ov::element::Type_t::u4:
        return "U4";
    case ov::element::Type_t::u8:
        return "U8";
    case ov::element::Type_t::u16:
        return "U16";
    case ov::element::Type_t::u32:
        return "U32";
    case ov::element::Type_t::u64:
        return "U64";
    case ov::element::Type_t::u1:
        return "BIN";
    case ov::element::Type_t::u2:
        return "U2";
    case ov::element::Type_t::boolean:
        return "BOOL";
    case ov::element::Type_t::dynamic:
        return "DYNAMIC";
    default:
        OPENVINO_THROW("Incorrect precision: ", precision);
    }
}

/**
 * @brief Gives the string representation of the default legacy layout value corresponding to the given rank.
 * @details This is done in order to assure the backward compatibility with the driver. Giving a layout different from
 * the default one may lead either to error or to accuracy failures since unwanted transposition layers may be
 * introduced.
 */
std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "**SCALAR**";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

bool isInitMetadata(const intel_npu::NetworkMetadata& networkMetadata) {
    if (networkMetadata.inputs.size() == 0) {
        return false;
    }
    return networkMetadata.inputs.at(0).isInitInputWeights;
}

/**
 * @brief Stores the information within the "WeightlessCacheAttribute" as runtime fields that persist upon
 * serialization.
 * @details Constant nodes (weights) may contain as medatadata the "WeightlessCacheAttribute", that is information
 * regarding the offset of the weights within the binary file, as well as the original size and precision. This
 * information is required within the "weights separation" flow, therefore this function is here to store it.
 * @note Not calling this function in the weights separation flow would lead to this information being lost upon
 * serialization. The "WeightlessCacheAttribute" information that is populated upon de-serialization would represent
 * metadata corresponding to the serialized stream, not the original weights file. Therefore the compiler would be
 * misinformed and lookups of weights offsets could fail.
 *
 * @param model Both source and target.
 */
void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model) {
    size_t constantId = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ov::RTMap& runtimeInfoMap = node->get_rt_info();
            const auto& weightlessCacheAttrIt =
                runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

            const std::string constantIdString = std::to_string(constantId++);
            if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                model->set_rt_info(weightlessCacheAttr.bin_offset, "ws_bin_offset_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_size, "ws_original_size_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_dtype, "ws_original_dtype_" + constantIdString);
            }
        }
    }
}

}  // namespace

namespace intel_npu {

DriverCompilerAdapter::DriverCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("DriverCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize DriverCompilerAdapter start");

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _compilerProperties = _zeroInitStruct->getCompilerProperties();

    _logger.info("DriverCompilerAdapter creating adapter using graphExtVersion");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    _logger.info("initialize DriverCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compile");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ov::pass::WeightsMapWrapper weightsMapWrapper;
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion, &weightsMapWrapper);

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    _logger.debug("build flags");
    buildFlags += serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += serializeConfig(config, compilerVersion);

    _logger.debug("compileIR Build flags : %s", buildFlags.c_str());

    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty() || config.get<BYPASS_UMD_CACHING>()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    _logger.debug("compile start");
    auto graphDesc = _zeGraphExt->getGraphDescriptor(std::move(serializedIR), buildFlags, flags);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphDesc);
    networkMeta.name = model->get_friendly_name();

    return std::make_shared<Graph>(_zeGraphExt,
                                   _zeroInitStruct,
                                   graphDesc,
                                   std::move(networkMeta),
                                   /* blob = */ std::nullopt,
                                   config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                         const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "compileWS");

    storeWeightlessCacheAttribute(model);

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    if ((compilerVersion.major < 6) || (compilerVersion.major == 6 && compilerVersion.minor < 3)) {
        OPENVINO_THROW("Minimum compiler version required for weights separation: 6.3. Found: ",
                       compilerVersion.major,
                       ".",
                       compilerVersion.minor);
    }

    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    if (config.get<SEPARATE_WEIGHTS_VERSION>() != ov::intel_npu::WSVersion::ITERATIVE) {
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call:",
                       config.get<SEPARATE_WEIGHTS_VERSION>(),
                       ". \"WSVersion::ITERATIVE\" is the only supported value for the compiler-in-driver path.");
    }

    _logger.debug("serialize IR");
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

    std::string buildFlags;
    const bool useIndices = !((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 9));

    const std::string serializedIOInfo = serializeIOInfo(model, useIndices);
    const FilteredConfig* plgConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (plgConfig == nullptr) {
        OPENVINO_THROW("config is not FilteredConfig");
    }
    FilteredConfig updatedConfig = *plgConfig;

    // If UMD Caching is requested to be bypassed or if OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty() || config.get<BYPASS_UMD_CACHING>()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    // WS v3 is based on a stateless compiler. We'll use a separate config entry for informing the compiler the index of
    // the current call iteration.
    std::vector<NetworkMetadata> initNetworkMetadata;
    NetworkMetadata mainNetworkMetadata;
    std::vector<GraphDescriptor> initGraphDescriptors;
    GraphDescriptor mainGraphHandle;
    size_t callNumber = 0;

    // Convention: run until the main schedule has been returned.
    int64_t compile_model_mem_start = 0;
    if (_logger.level() >= ov::log::Level::INFO) {
        compile_model_mem_start = get_peak_memory_usage();
    }
    while (true) {
        _logger.debug("compileWS iteration %d", callNumber);
        updatedConfig.update({{ov::intel_npu::ws_compile_call_number.name(), std::to_string(callNumber++)}});

        _logger.debug("build flags");
        buildFlags = serializedIOInfo;
        buildFlags += " ";
        buildFlags += serializeConfig(updatedConfig, compilerVersion);

        _logger.debug("compile start");
        auto graphDesc = _zeGraphExt->getGraphDescriptor(serializedIR, buildFlags, flags);
        _logger.debug("compile end");

        OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
        NetworkMetadata networkMetadata = _zeGraphExt->getNetworkMeta(graphDesc);

        if (isInitMetadata(networkMetadata)) {
            networkMetadata.name = model->get_friendly_name() + "_init";
            initNetworkMetadata.push_back(std::move(networkMetadata));
            initGraphDescriptors.push_back(graphDesc);
        } else {
            networkMetadata.name = model->get_friendly_name() + "_main";
            mainNetworkMetadata = std::move(networkMetadata);
            mainGraphHandle = graphDesc;
            serializedIR = SerializedIR();
            // By convention, the main schedule is the last result produced by the compiler
            break;
        }
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        auto compile_model_mem_end = get_peak_memory_usage();
        _logger.debug("Start of compilation memory usage: Peak %lld KB", compile_model_mem_start);
        _logger.debug("End of compilation memory usage: Peak %lld KB", compile_model_mem_end);
        // Note: Following log is parsed by CI. Take care when modifying it.
        _logger.info("Compilation memory usage: Peak %lld KB", compile_model_mem_end - compile_model_mem_start);
    }

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphHandle,
                                             std::move(mainNetworkMetadata),
                                             /* mainBlob = */ std::nullopt,
                                             initGraphDescriptors,
                                             std::move(initNetworkMetadata),
                                             /* initBlobs = */ std::nullopt,
                                             model,
                                             config);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::parse(
    ov::Tensor mainBlob,
    const Config& config,
    std::optional<std::vector<ov::Tensor>> initBlobs,
    const std::optional<std::shared_ptr<const ov::Model>>& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto mainGraphDesc = _zeGraphExt->getGraphDescriptor(mainBlob.data(), mainBlob.get_byte_size());
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(mainGraphDesc);

    // exporting the blob when we get it from cache or ov::hint::compiled_blob property
    // shall be available
    const bool blobIsPersistent = config.has<COMPILED_BLOB>()       ? true
                                  : config.has<LOADED_FROM_CACHE>() ? config.get<LOADED_FROM_CACHE>()
                                                                    : false;

    if (!initBlobs.has_value()) {
        return std::make_shared<Graph>(_zeGraphExt,
                                       _zeroInitStruct,
                                       mainGraphDesc,
                                       std::move(networkMeta),
                                       std::move(mainBlob),
                                       config,
                                       blobIsPersistent);
    }

    // The presence of init schedules means weights separation has been enabled at compilation time. Use a specific
    // "Graph" object as wrapper over all L0 handles.
    std::vector<GraphDescriptor> initGraphDescriptors;
    std::vector<NetworkMetadata> initMetadata;

    for (const auto& initBlob : initBlobs.value()) {
        auto initGraphDesc = _zeGraphExt->getGraphDescriptor(initBlob.data(), initBlob.get_byte_size());

        initGraphDescriptors.push_back(initGraphDesc);
        initMetadata.push_back(_zeGraphExt->getNetworkMeta(initGraphDesc));
    }

    return std::make_shared<WeightlessGraph>(_zeGraphExt,
                                             _zeroInitStruct,
                                             mainGraphDesc,
                                             std::move(networkMeta),
                                             std::move(mainBlob),
                                             initGraphDescriptors,
                                             std::move(initMetadata),
                                             std::move(initBlobs),
                                             model.value(),
                                             config,
                                             blobIsPersistent);
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    ov::pass::WeightsMapWrapper weightsMapWrapper;
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion, &weightsMapWrapper);

    std::string buildFlags;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("queryImpl build flags : %s", buildFlags.c_str());

    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    try {
        const auto supportedLayers = _zeGraphExt->queryGraph(std::move(serializedIR), buildFlags);
        for (auto&& layerName : supportedLayers) {
            result.emplace(layerName, deviceName);
        }
        _logger.info("For given model, there are %d supported layers", supportedLayers.size());
    } catch (std::exception& e) {
        OPENVINO_THROW("Fail in calling querynetwork : ", e.what());
    }

    _logger.debug("query end");
    return result;
}

uint32_t DriverCompilerAdapter::get_version() const {
    return _zeroInitStruct->getCompilerVersion();
}

/**
 * @brief Place xml + weights in sequential memory
 * @details Format of the memory:
 */
SerializedIR DriverCompilerAdapter::serializeIR(const std::shared_ptr<const ov::Model>& model,
                                                ze_graph_compiler_version_info_t compilerVersion,
                                                const uint32_t supportedOpsetVersion,
                                                ov::pass::WeightsMapWrapper* weightsMapWrapper) const {
    // TODO: version check to see if we use traditional or new way since old driver can not parse map
    bool useWeightsMap = (compilerVersion.major > 7) || (compilerVersion.major == 7 && compilerVersion.minor >= 22);
    if (weightsMapWrapper == nullptr) {
        useWeightsMap = false;
    }

    driver_compiler_utils::IRSerializer irSerializer(model,
                                                     supportedOpsetVersion,
                                                     useWeightsMap ? weightsMapWrapper : nullptr);

    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(irSerializer.getXmlSize());
    const uint64_t weightsSize = static_cast<uint64_t>(irSerializer.getWeightsSize());

    OPENVINO_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        OPENVINO_THROW("Xml file is too big to process. xmlSize: ", xmlSize, " >= maxSizeOfXML: ", maxSizeOfXML);
    }
    if (weightsSize >= maxSizeOfWeights) {
        OPENVINO_THROW("Bin file is too big to process. xmlSize: ",
                       weightsSize,
                       " >= maxSizeOfWeights: ",
                       maxSizeOfWeights);
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xmlSize + sizeof(weightsSize) + weightsSize;

    // use array to avoid vector's memory zeroing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    uint64_t offset = 0;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    // xml data is filled in serializeModel()
    uint64_t xmlOffset = offset;
    offset += xmlSize;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    // weights data is filled in serializeModel()
    uint64_t weightsOffset = offset;
    offset += weightsSize;

    irSerializer.serializeModelToBuffer(serializedIR + xmlOffset, serializedIR + weightsOffset);

    OPENVINO_ASSERT(offset == sizeOfSerializedIR);

    return std::make_pair(sizeOfSerializedIR, buffer);
}

std::string DriverCompilerAdapter::serializeIOInfo(const std::shared_ptr<const ov::Model>& model,
                                                   const bool useIndices) const {
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;
    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    inputsPrecisionSS << INPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    inputsLayoutSS << INPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    const auto getRankOrThrow = [](const ov::PartialShape& shape) -> size_t {
        if (shape.rank().is_dynamic()) {
            OPENVINO_THROW("Dynamic rank is not supported for NPU plugin");
        }
        return shape.rank().get_length();
    };

    if (!parameters.empty()) {
        size_t parameterIndex = 0;

        for (const std::shared_ptr<ov::op::v0::Parameter>& parameter : parameters) {
            const auto precision = parameter->get_element_type();
            const auto rank = getRankOrThrow(parameter->get_partial_shape());

            if (parameterIndex != 0) {
                inputsPrecisionSS << VALUES_SEPARATOR;
                inputsLayoutSS << VALUES_SEPARATOR;
            }

            if (useIndices) {
                inputsPrecisionSS << parameterIndex;
                inputsLayoutSS << parameterIndex;
            } else {
                const std::string& name = parameter->get_friendly_name();

                inputsPrecisionSS << name;
                // Ticket: E-88902
                inputsLayoutSS << name;
            }

            inputsPrecisionSS << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
            inputsLayoutSS << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);

            ++parameterIndex;
        }
    }

    inputsPrecisionSS << VALUE_DELIMITER;
    inputsLayoutSS << VALUE_DELIMITER;

    outputsPrecisionSS << OUTPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    outputsLayoutSS << OUTPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    size_t resultIndex = 0;
    for (const std::shared_ptr<ov::op::v0::Result>& result : results) {
        const auto precision = result->get_element_type();
        const auto rank = getRankOrThrow(result->get_output_partial_shape(0));

        if (resultIndex != 0) {
            outputsPrecisionSS << VALUES_SEPARATOR;
            outputsLayoutSS << VALUES_SEPARATOR;
        }

        if (useIndices) {
            outputsPrecisionSS << resultIndex;
            outputsLayoutSS << resultIndex;
        } else {
            const std::string& name = result->get_input_node_ptr(0)->get_friendly_name();

            outputsPrecisionSS << name;
            outputsLayoutSS << name;
        }

        outputsPrecisionSS << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
        outputsLayoutSS << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);

        ++resultIndex;
    }

    outputsPrecisionSS << VALUE_DELIMITER;
    outputsLayoutSS << VALUE_DELIMITER;

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + VALUES_SEPARATOR.data() + inputsLayoutSS.str() + VALUES_SEPARATOR.data() +
           outputsPrecisionSS.str() + VALUES_SEPARATOR.data() + outputsLayoutSS.str();
}

std::string DriverCompilerAdapter::serializeConfig(const Config& config,
                                                   ze_graph_compiler_version_info_t compilerVersion) const {
    Logger logger("serializeConfig", Logger::global().level());

    std::string content = {};

    const FilteredConfig* plgConfig = dynamic_cast<const FilteredConfig*>(&config);
    if (plgConfig != nullptr) {
        content += plgConfig->toStringForCompiler();
        content += plgConfig->toStringForCompilerInternal();
    } else {
        logger.warning("Failed to cast Config to FilteredConfig. Exporting all configs");
        content += config.toString();
    }

    logger.debug("Original content of config: %s", content.c_str());

    // Remove optimization-level and performance-hint-override for old driver which not support them
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 7)) {
        std::string valueOfParams = config.get<COMPILATION_MODE_PARAMS>();
        std::string keyOfOptL("optimization-level");
        std::string keyOfPerfHO("performance-hint-override");
        if (valueOfParams != "" && (valueOfParams.find(keyOfOptL) != std::string::npos ||
                                    valueOfParams.find(keyOfPerfHO) != std::string::npos)) {
            // Remove unsupported options from value
            std::ostringstream optLevelStr;
            optLevelStr << keyOfOptL << KEY_VALUE_SEPARATOR << "\\d+";
            std::ostringstream perfHintStr;
            perfHintStr << keyOfPerfHO << KEY_VALUE_SEPARATOR << "\\S+";
            logger.warning("%s property is not supported by this compiler version. Removing from parameters",
                           keyOfOptL.c_str());
            valueOfParams = std::regex_replace(valueOfParams, std::regex(optLevelStr.str()), "");
            logger.warning("%s property is not supported by this compiler version. Removing from parameters",
                           keyOfPerfHO.c_str());
            valueOfParams = std::regex_replace(valueOfParams, std::regex(perfHintStr.str()), "");

            // Trim space
            valueOfParams = std::regex_replace(valueOfParams, std::regex(R"(^\s+|\s+$)"), "");

            // Replace the value in content with new value
            std::ostringstream compilationParamsStr;
            compilationParamsStr << ov::intel_npu::compilation_mode_params.name() << KEY_VALUE_SEPARATOR
                                 << VALUE_DELIMITER << ".*" << VALUE_DELIMITER;
            if (valueOfParams == "") {
                logger.warning("Clear empty NPU_COMPILATION_MODE_PARAMS. Removing from parameters");
                content = std::regex_replace(content, std::regex(compilationParamsStr.str()), "");
            } else {
                std::ostringstream newValue;
                newValue << ov::intel_npu::compilation_mode_params.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                         << valueOfParams << VALUE_DELIMITER;
                logger.warning("Replace value of NPU_COMPILATION_MODE_PARAMS with new value %s",
                               newValue.str().c_str());
                content = std::regex_replace(content, std::regex(compilationParamsStr.str()), newValue.str().c_str());
            }
        }
    }

    // As a consequence of complying to the conventions established in the 2.0 OV API, the set of values corresponding
    // to the "model priority" key has been modified cpu_pinning property is not supported in compilers < v5.2 - need to
    // remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 2)) {
        const auto& getTargetRegex = [](const ov::hint::Priority& priorityValue) -> std::regex {
            std::ostringstream result;
            result << ov::hint::model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << priorityValue
                   << VALUE_DELIMITER;
            return std::regex(result.str());
        };
        const auto& getStringReplacement = [](const ov::intel_npu::LegacyPriority& priorityValue) -> std::string {
            std::ostringstream result;
            result << ov::intel_npu::legacy_model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                   << priorityValue << VALUE_DELIMITER;
            return result.str();
        };

        // E.g. (valid as of writing this): MODEL_PRIORITY="MEDIUM" -> MODEL_PRIORITY="MODEL_PRIORITY_MED"
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::LOW),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::LOW));
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::MEDIUM),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::MEDIUM));
        content = std::regex_replace(content,
                                     getTargetRegex(ov::hint::Priority::HIGH),
                                     getStringReplacement(ov::intel_npu::LegacyPriority::HIGH));
    }

    // Special case for compiler Turbo
    // NPU_TURBO is a special option in the sense that by default it is a driver-setting, but certain compilers support
    // and make use of it too If we have turbo in the config string, we check if compiler supports it. If it doesn't
    // support it, we remove it
    if (std::regex_search(content, std::regex("NPU_TURBO"))) {
        bool is_supported = false;
        try {
            is_supported = is_option_supported("NPU_TURBO");
        } catch (...) {
            // mute it, not critical
            is_supported = false;
        }
        if (!is_supported) {
            std::ostringstream turbostr;
            turbostr << ov::intel_npu::turbo.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                     << VALUE_DELIMITER;
            logger.info("NPU_TURBO property is not supported by this compiler. Removing from "
                        "parameters");
            content = std::regex_replace(content, std::regex(turbostr.str()), "");
        }
    }

    // FINAL step to convert prefixes of remaining params, to ensure backwards compatibility
    // From 5.0.0, driver compiler start to use NPU_ prefix, the old version uses VPU_ prefix
    if (compilerVersion.major < 5) {
        std::regex reg("NPU_");
        content = std::regex_replace(content, reg, "VPU_");
        // From 4.0.0, driver compiler start to use VPU_ prefix, the old version uses VPUX_ prefix
        if (compilerVersion.major < 4) {
            // Replace VPU_ with VPUX_ for old driver compiler
            std::regex reg("VPU_");
            content = std::regex_replace(content, reg, "VPUX_");
        }
    }

    return "--config " + content;
}

std::vector<std::string> DriverCompilerAdapter::get_supported_options() const {
    std::string compilerOptionsStr;
    compilerOptionsStr = _zeGraphExt->getCompilerSupportedOptions();
    // vectorize string
    std::istringstream suppstream(compilerOptionsStr);
    std::vector<std::string> compilerOpts;
    std::string option;
    while (suppstream >> option) {
        compilerOpts.push_back(option);
    }
    return compilerOpts;
}

bool DriverCompilerAdapter::is_option_supported(std::string optname) const {
    return _zeGraphExt->isOptionSupported(std::move(optname));
}

}  // namespace intel_npu
