// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_serializer.hpp"

#include <cstdint>
#include <istream>
#include <streambuf>

#include "intel_npu/xml_serializer.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

namespace {

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

}  // namespace

namespace intel_npu::driver_compiler_utils {

IRSerializerBase::IRSerializerBase(const std::shared_ptr<const ov::Model>& origModel,
                                   const ze_graph_compiler_version_info_t compilerVersion,
                                   const uint32_t supportedOpset)
    : _logger("IRSerializerBase", Logger::global().level()),
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

IRSerializerWithWeightsCopy::IRSerializerWithWeightsCopy(const std::shared_ptr<const ov::Model>& origModel,
                                                         const ze_graph_compiler_version_info_t compilerVersion,
                                                         const uint32_t supportedOpset)
    : IRSerializerBase(origModel, compilerVersion, supportedOpset) {
    _logger.setName("IRSerializerWithWeightsCopy");
};

IRSerializerWithoutWeightsCopy::IRSerializerWithoutWeightsCopy(const std::shared_ptr<const ov::Model>& origModel,
                                                               const ze_graph_compiler_version_info_t compilerVersion,
                                                               const uint32_t supportedOpset)
    : IRSerializerBase(origModel, compilerVersion, supportedOpset) {
    _logger.setName("IRSerializerWithoutWeightsCopy");
};

void IRSerializerWithWeightsCopy::serializeModelToStream(std::ostream& xml, std::ostream& weights) {
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

void IRSerializerWithoutWeightsCopy::serializeModelToStream(std::ostream& stream) {
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

void IRSerializerWithWeightsCopy::countModelSize() {
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

void IRSerializerWithoutWeightsCopy::countModelSize() {
    _logger.debug("countModelSize");

    counter_streambuf streamBuf;
    std::ostream stream(&streamBuf);

    serializeModelToStream(stream);

    _serializedModelSize = streamBuf.size();

    _logger.debug("countModelSize completed, serialized model size: %d", _serializedModelSize);
}

void IRSerializerWithWeightsCopy::serializeModelToBuffer(uint8_t* xml, uint8_t* weights) {
    _logger.debug("serializeModelToBuffer");

    writer_streambuf xmlStreamBuf(xml);
    writer_streambuf weightsStreamBuf(weights);
    std::ostream xmlStream(&xmlStreamBuf);
    std::ostream weightsStream(&weightsStreamBuf);

    serializeModelToStream(xmlStream, weightsStream);

    _logger.debug("serializeModelToBuffer end");
}

void IRSerializerWithoutWeightsCopy::serializeModelToBuffer(uint8_t* buffer) {
    _logger.debug("serializeModelToBuffer");

    writer_streambuf streamBuf(buffer);
    std::ostream stream(&streamBuf);

    serializeModelToStream(stream);

    _logger.debug("serializeModelToBuffer end");
}

SerializedIR IRSerializerWithWeightsCopy::serialize() {
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

SerializedIR IRSerializerWithoutWeightsCopy::serialize() {
    countModelSize();

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

}  // namespace intel_npu::driver_compiler_utils
