// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_serializer.hpp"

#include <chrono>
#include <cstdint>
#include <istream>
#include <mutex>
#include <regex>
#include <streambuf>

#include "custom_stream_buffer.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/weights_pointer_attribute.hpp"
#include "intel_npu/xml_serializer.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

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

void storeWeightsPointerAttribute(const std::shared_ptr<ov::Model>& model, const size_t weightSizeThreshold) {
    for (auto&& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        if (constantNode->get_byte_size() < weightSizeThreshold) {
            continue;
        }

        ov::RTMap& runtimeInfoMap = constantNode->get_rt_info();
        runtimeInfoMap[intel_npu::WeightsPointerAttribute::get_type_info_static()] =
            intel_npu::WeightsPointerAttribute(constantNode->get_data_ptr(), constantNode->get_byte_size());
    }
}

void removeWeightsPointerAttribute(const std::shared_ptr<ov::Model>& model) {
    for (auto&& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        ov::RTMap& runtimeInfoMap = node->get_rt_info();
        const auto& resultIt = runtimeInfoMap.find(intel_npu::WeightsPointerAttribute::get_type_info_static());
        if (resultIt != runtimeInfoMap.end()) {
            runtimeInfoMap.erase(resultIt);
        }
    }
}

}  // namespace

namespace intel_npu::driver_compiler_utils {

VCLSerializerBase::VCLSerializerBase(const std::shared_ptr<const ov::Model>& origModel,
                                     const ze_graph_compiler_version_info_t compilerVersion,
                                     const uint32_t supportedOpset)
    : _logger("VCLSerializerBase", Logger::global().level()),
      _compilerVersion(compilerVersion),
      _supportedOpset(supportedOpset) {
    // There is no const variant of run_passes so use const_cast here
    // as model serialization does not mutate the model
    _model = std::const_pointer_cast<ov::Model>(origModel);

    if (supportedOpset < 11) {
        // Need to clone to modify the model and remain thread safe
        _model = _model->clone();
        _logger.info("Clone model for offset smaller than 11");
    }
}

VCLSerializerWithWeightsCopy::VCLSerializerWithWeightsCopy(const std::shared_ptr<const ov::Model>& origModel,
                                                           const ze_graph_compiler_version_info_t compilerVersion,
                                                           const uint32_t supportedOpset)
    : VCLSerializerBase(origModel, compilerVersion, supportedOpset) {
    _logger.setName("VCLSerializerWithWeightsCopy");
};

VCLSerializerWithoutWeightsCopy::VCLSerializerWithoutWeightsCopy(const std::shared_ptr<const ov::Model>& origModel,
                                                                 const ze_graph_compiler_version_info_t compilerVersion,
                                                                 const uint32_t supportedOpset,
                                                                 const size_t weightsSizeThreshold)
    : VCLSerializerBase(origModel, compilerVersion, supportedOpset),
      _weightsSizeThreshold(weightsSizeThreshold) {
    _logger.setName("VCLSerializerWithoutWeightsCopy");
};

void VCLSerializerWithWeightsCopy::serializeModelToStream(std::ostream& xml, std::ostream& weights) {
    _logger.debug("serializeModelToStream");
    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(std::move(passConfig), "NPU:serializeModelToStream");

    if (_supportedOpset < 11) {
        // Downgrade to opset10
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
        _logger.info("Downgrade op for opset smaller than 11");
    }

    manager.register_pass<ov::pass::Serialize>(xml, weights);

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    const auto newAPIKey = "is_new_api";

    // Flag used for indicating an NPU plugin version which switched the I/O identification convention from names to
    // indices. The flag is required in order to inform the driver-compiler adapter to expect indices when attempting to
    // deserialize the I/O metadata.
    const auto useIndicesForIOMetadata = "use_indices_for_io_metadata";

    // We modify the original model object here therefore a mutex is required
    {
        std::lock_guard<std::mutex> lock(rtInfoMutex);

        _model->set_rt_info(true, newAPIKey);
        _model->set_rt_info(true, useIndicesForIOMetadata);

        manager.run_passes(_model);

        auto& rtInfo = _model->get_rt_info();
        rtInfo.erase(newAPIKey);
        rtInfo.erase(useIndicesForIOMetadata);
    }
    _logger.debug("serializeModelToStream end");
}

void VCLSerializerWithoutWeightsCopy::serializeModelToStream(std::ostream& stream) {
    _logger.debug("serializeModelToStream");
    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(std::move(passConfig), "NPU:serializeModelToStream");

    if (_supportedOpset < 11) {
        // Downgrade to opset10
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
        _logger.info("Downgrade op for opset smaller than 11");
    }
    manager.register_pass<StreamSerialize>(stream);

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    const auto newAPIKey = "is_new_api";

    // Flag used for indicating an NPU plugin version which switched the I/O identification convention from names to
    // indices. The flag is required in order to inform the driver-compiler adapter to expect indices when attempting to
    // deserialize the I/O metadata.
    const auto useIndicesForIOMetadata = "use_indices_for_io_metadata";

    // We modify the original model object here therefore a mutex is required
    {
        std::lock_guard<std::mutex> lock(rtInfoMutex);

        _model->set_rt_info(true, newAPIKey);
        _model->set_rt_info(true, useIndicesForIOMetadata);

        manager.run_passes(_model);

        auto& rtInfo = _model->get_rt_info();
        rtInfo.erase(newAPIKey);
        rtInfo.erase(useIndicesForIOMetadata);
    }
    _logger.debug("serializeModelToStream end");
}

void VCLSerializerWithWeightsCopy::countModelSize() {
    _logger.debug("countModelSize");

    counter_streambuf xmlStreamBuf;
    counter_streambuf weightsStreamBuf;
    std::ostream xmlStream(&xmlStreamBuf);
    std::ostream weightsStream(&weightsStreamBuf);

    serializeModelToStream(xmlStream, weightsStream);

    _xmlSize = xmlStreamBuf.size();
    _weightsSize = weightsStreamBuf.size();

    _logger.debug("countModelSize completed, xml size: %d, weights size: %d", _xmlSize, _weightsSize);
}

void VCLSerializerWithoutWeightsCopy::countModelSize() {
    _logger.debug("countModelSize");

    counter_streambuf streamBuf;
    std::ostream stream(&streamBuf);

    serializeModelToStream(stream);

    _serializedModelSize = streamBuf.size();

    _logger.debug("countModelSize completed, serialized model size: %d", _serializedModelSize);
}

void VCLSerializerWithWeightsCopy::serializeModelToBuffer(uint8_t* xml, uint8_t* weights) {
    _logger.debug("serializeModelToBuffer");

    writer_streambuf xmlStreamBuf(xml);
    writer_streambuf weightsStreamBuf(weights);
    std::ostream xmlStream(&xmlStreamBuf);
    std::ostream weightsStream(&weightsStreamBuf);

    serializeModelToStream(xmlStream, weightsStream);

    _logger.debug("serializeModelToBuffer end");
}

void VCLSerializerWithoutWeightsCopy::serializeModelToBuffer(uint8_t* buffer) {
    _logger.debug("serializeModelToBuffer");

    writer_streambuf streamBuf(buffer);
    std::ostream stream(&streamBuf);

    serializeModelToStream(stream);

    _logger.debug("serializeModelToBuffer end");
}

SerializedIR VCLSerializerWithWeightsCopy::serialize() {
    countModelSize();

    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(_xmlSize);
    const uint64_t weightsSize = static_cast<uint64_t>(_weightsSize);

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

    const uint64_t sizeOfSerializedIR = sizeof(_compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xmlSize + sizeof(weightsSize) + weightsSize;

    // use array to avoid vector's memory zeroing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    uint64_t offset = 0;
    checkedMemcpy(serializedIR + offset, sizeOfSerializedIR - offset, &_compilerVersion, sizeof(_compilerVersion));
    offset += sizeof(_compilerVersion);

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

    serializeModelToBuffer(serializedIR + xmlOffset, serializedIR + weightsOffset);

    OPENVINO_ASSERT(offset == sizeOfSerializedIR);

    return std::make_pair(sizeOfSerializedIR, buffer);
}

SerializedIR VCLSerializerWithoutWeightsCopy::serialize() {
    countModelSize();  // TODO refactor, we don't need this

    if (_serializedModelSize >= std::numeric_limits<uint64_t>::max()) {
        OPENVINO_THROW("The serialized model is too big to process. Size: ",
                       _serializedModelSize,
                       " >= ",
                       std::numeric_limits<uint64_t>::max());
    }

    const uint64_t sizeOfSerializedIR = sizeof(_compilerVersion) + sizeof(_serializedModelSize) + _serializedModelSize;

    // use array to avoid vector's memory zero-ing overhead
    std::shared_ptr<uint8_t> buffer(new uint8_t[sizeOfSerializedIR], std::default_delete<uint8_t[]>());
    uint8_t* serializedIR = buffer.get();

    checkedMemcpy(serializedIR, sizeOfSerializedIR, &_compilerVersion, sizeof(_compilerVersion));

    uint64_t offset = sizeof(_compilerVersion);
    checkedMemcpy(serializedIR + offset,
                  sizeOfSerializedIR - offset,
                  &_serializedModelSize,
                  sizeof(_serializedModelSize));
    offset += sizeof(_serializedModelSize);

    serializeModelToBuffer(serializedIR + offset);

    OPENVINO_ASSERT(offset + _serializedModelSize == sizeOfSerializedIR);

    return SerializedIR(sizeOfSerializedIR, buffer);
}

SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                         const ze_graph_compiler_version_info_t compilerVersion,
                         const uint32_t supportedOpsetVersion,
                         const bool useBaseModelSerializer,
                         const size_t weightsSizeThreshold) {
    if (!useBaseModelSerializer) {
        // Required for adding & removing weights pointer attributes
        const std::shared_ptr<ov::Model> nonConstantModel = std::const_pointer_cast<ov::Model>(model);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        storeWeightsPointerAttribute(nonConstantModel, weightsSizeThreshold);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time to store attributes: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

        SerializedIR serializedIR =
            VCLSerializerWithoutWeightsCopy(model, compilerVersion, supportedOpsetVersion, weightsSizeThreshold)
                .serialize();
        removeWeightsPointerAttribute(nonConstantModel);
        return serializedIR;
    }
    return VCLSerializerWithWeightsCopy(model, compilerVersion, supportedOpsetVersion).serialize();
}

std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices) {
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

std::string serializeConfig(const Config& config,
                            ze_graph_compiler_version_info_t compilerVersion,
                            bool turboSupported) {
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
            is_supported = turboSupported;
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

}  // namespace intel_npu::driver_compiler_utils
