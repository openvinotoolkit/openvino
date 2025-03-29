// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include <regex>
#include <string_view>

#include "driver_graph.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "ir_serializer.hpp"
#include "openvino/core/model.hpp"

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
void checkedMemcpy(void* destination, size_t destinationSize, void const* source, size_t numberOfBytes) {
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
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

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
    ze_graph_handle_t graphHandle = _zeGraphExt->getGraphHandle(std::move(serializedIR), buildFlags, flags);
    _logger.debug("compile end");

    OV_ITT_TASK_NEXT(COMPILE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphHandle);
    networkMeta.name = model->get_friendly_name();

    return std::make_shared<DriverGraph>(_zeGraphExt,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkMeta),
                                         config,
                                         nullptr);
}

std::shared_ptr<IGraph> DriverCompilerAdapter::parse(std::unique_ptr<BlobContainer> blobPtr,
                                                     const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "parse");

    _logger.debug("parse start");
    ze_graph_handle_t graphHandle =
        _zeGraphExt->getGraphHandle(*reinterpret_cast<const uint8_t*>(blobPtr->get_ptr()), blobPtr->size());
    _logger.debug("parse end");

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    auto networkMeta = _zeGraphExt->getNetworkMeta(graphHandle);

    return std::make_shared<DriverGraph>(_zeGraphExt,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkMeta),
                                         config,
                                         std::move(blobPtr));
}

ov::SupportedOpsMap DriverCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(query_BLOB, itt::domains::NPUPlugin, "DriverCompilerAdapter", "query");

    const ze_graph_compiler_version_info_t& compilerVersion = _compilerProperties.compilerVersion;
    const auto maxOpsetVersion = _compilerProperties.maxOVOpsetVersionSupported;
    _logger.info("getSupportedOpsetVersion Max supported version of opset in CiD: %d", maxOpsetVersion);

    _logger.debug("serialize IR");
    auto serializedIR = serializeIR(model, compilerVersion, maxOpsetVersion);

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
                                                const uint32_t supportedOpsetVersion) const {
    driver_compiler_utils::IRSerializer irSerializer(model, supportedOpsetVersion);

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

    std::string content = config.toString();

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

        // Removing cpu_pinning from the command string
        std::ostringstream pinningstr;
        pinningstr << ov::hint::enable_cpu_pinning.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                   << VALUE_DELIMITER;
        logger.warning(
            "ENABLE_CPU_PINNING property is not supported by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(pinningstr.str()), "");
    }

    /// Stepping and max_tiles are not supported in versions < 5.3 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 3)) {
        std::ostringstream stepstr;
        stepstr << ov::intel_npu::stepping.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\d+"
                << VALUE_DELIMITER;
        std::ostringstream maxtilestr;
        maxtilestr << ov::intel_npu::max_tiles.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\d+"
                   << VALUE_DELIMITER;
        logger.warning("NPU_STEPPING property is not supported by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(stepstr.str()), "");
        logger.warning("NPU_MAX_TILES property is not supported by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(maxtilestr.str()), "");
    }

    /// Removing INFERENCE_PRECISION_HINT for older compilers
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 4)) {
        std::ostringstream precstr;
        precstr << ov::hint::inference_precision.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                << VALUE_DELIMITER;
        logger.warning(
            "INFERENCE_PRECISION_HINT property is not supported by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(precstr.str()), "");
    }

    /// Replacing NPU_TILES (for all versions) with NPU_DPU_GROUPS for backwards compatibility
    if (std::regex_search(content, std::regex(ov::intel_npu::tiles.name()))) {
        logger.warning("NPU_TILES property is not supported by this compiler version. Swaping it to "
                       "NPU_DPU_GROUPS (obsolete)");
        content = std::regex_replace(content, std::regex(ov::intel_npu::tiles.name()), "NPU_DPU_GROUPS");
    }

    // Batch mode property is not supported in versions < 5.5 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 5)) {
        std::ostringstream batchstr;
        batchstr << ov::intel_npu::batch_mode.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                 << VALUE_DELIMITER;

        logger.warning("NPU_BATCH_MODE property is not supported by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(batchstr.str()), "");
    }

    // EXECUTION_MODE_HINT is not supported in versions < 5.6 - need to remove it
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 6)) {
        std::ostringstream batchstr;
        batchstr << ov::hint::execution_mode.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                 << VALUE_DELIMITER;
        logger.warning(
            "EXECUTION_MODE_HINT property is not supported by this compiler version. Removing from parameters");
        content = std::regex_replace(content, std::regex(batchstr.str()), "");
    }

    // COMPILER_DYNAMIC_QUANTIZATION is not supported in versions < 7.1 - need to remove it
    if ((compilerVersion.major < 7) || (compilerVersion.major == 7 && compilerVersion.minor < 1)) {
        std::ostringstream dqstr;
        dqstr << ov::intel_npu::compiler_dynamic_quantization.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
              << VALUE_DELIMITER;
        logger.warning(
            "COMPILER_DYNAMIC_QUANTIZATION property is not supported by this compiler version. Removing from "
            "parameters");
        content = std::regex_replace(content, std::regex(dqstr.str()), "");
    }

    // QDQ_OPTIMIZATION is not supported in versions < 7.20 - need to remove it
    if ((compilerVersion.major < 7) || (compilerVersion.major == 7 && compilerVersion.minor < 20)) {
        std::ostringstream qdqstr;
        qdqstr << ov::intel_npu::qdq_optimization.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
               << VALUE_DELIMITER;
        logger.warning("NPU_QDQ_OPTIMIZATION property is not supported by this compiler version. Removing from "
                       "parameters");
        content = std::regex_replace(content, std::regex(qdqstr.str()), "");
    }

    // BATCH_COMPILER_MODE_SETTINGS is not supported in versions < 7.4 - need to remove it
    if ((compilerVersion.major < 7) || (compilerVersion.major == 7 && compilerVersion.minor < 4)) {
        std::ostringstream dqstr;
        dqstr << ov::intel_npu::batch_compiler_mode_settings.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
              << VALUE_DELIMITER;
        logger.warning("BATCH_COMPILER_MODE_SETTINGS property is not supported by this compiler version. Removing from "
                       "parameters");
        content = std::regex_replace(content, std::regex(dqstr.str()), "");
    }

    // NPU_DEFER_WEIGHTS_LOAD is needed at runtime only
    {
        std::ostringstream batchstr;
        batchstr << ov::intel_npu::defer_weights_load.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                 << VALUE_DELIMITER;
        logger.info("NPU_DEFER_WEIGHTS_LOAD property is needed at runtime only. Removing from parameters");
        content = std::regex_replace(content, std::regex(batchstr.str()), "");
    }

    // NPU_RUN_INFERENCES_SEQUENTIALLY is needed at runtime only
    {
        std::ostringstream batchstr;
        batchstr << ov::intel_npu::run_inferences_sequentially.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                 << "\\S+" << VALUE_DELIMITER;
        logger.info("NPU_RUN_INFERENCES_SEQUENTIALLY property is needed at runtime only. Removing from parameters");
        content = std::regex_replace(content, std::regex(batchstr.str()), "");
    }

    // Remove the properties that are not used by the compiler WorkloadType is used only by compiled model
    std::ostringstream workloadtypestr;
    workloadtypestr << ov::workload_type.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+" << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(workloadtypestr.str()), "");
    // Remove turbo property as it is not used by compiler
    std::ostringstream turbostring;
    turbostring << ov::intel_npu::turbo.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+" << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(turbostring.str()), "");
    // Remove weights path property as it is not used by compiler
    std::ostringstream weightspathstream;
    weightspathstream << ov::weights_path.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+" << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(weightspathstream.str()), "");
    // Remove Bypass UMD Caching propery
    std::ostringstream umdcachestring;
    umdcachestring << ov::intel_npu::bypass_umd_caching.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                   << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(umdcachestring.str()), "");

    std::ostringstream skipversioncheck;
    skipversioncheck << ov::intel_npu::disable_version_check.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << "\\S+"
                     << VALUE_DELIMITER;
    content = std::regex_replace(content, std::regex(skipversioncheck.str()), "");

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

}  // namespace intel_npu
