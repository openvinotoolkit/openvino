// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_serializer.hpp"

#include <cstdint>
#include <fstream>
#include <istream>
#include <mutex>
#include <streambuf>

#include "openvino/pass/serialize.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

namespace intel_npu::driver_compiler_utils {

IRSerializer::IRSerializer(const std::shared_ptr<const ov::Model>& origModel,
                           const uint32_t supportedOpset,
                           bool serializeWeightsPtrToXml)
    : _logger("IRSerializer", Logger::global().level()),
      _supportedOpset(supportedOpset),
      _serializeWeightsPtrToXml(serializeWeightsPtrToXml) {
    // There is no const variant of run_passes so use const_cast here
    // as model serialization does not mutate the model
    _model = std::const_pointer_cast<ov::Model>(origModel);

    if (supportedOpset < 11) {
        // Need to clone to modify the model and remain thread safe
        _model = _model->clone();
        _logger.info("Clone model for offset smaller than 11");
    }

    if (_serializeWeightsPtrToXml) {
        // Serialize directly to avoid size counter
        std::stringstream xmlStream;
        std::ofstream weightsStream;
        serializeModelToStream(xmlStream, weightsStream);
        _xmlString = xmlStream.str();
    } else {
        countModelSize();
    }
}

void IRSerializer::serializeModelToStream(std::ostream& xml, std::ostream& weights) {
    _logger.debug("serializeModelToStream");
    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(std::move(passConfig), "NPU:serializeModelToStream");

    if (_supportedOpset < 11) {
        // Downgrade to opset10
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
        _logger.info("Downgrade op for opset smaller than 11");
    }

    if (!_serializeWeightsPtrToXml) {
        _logger.debug("using default serialization");
        // If weights map is not provided, we use the default serialization
        manager.register_pass<ov::pass::Serialize>(xml, weights);
    } else {
        _logger.debug("serialize weights pointer to xml");
        // If weights map is provided, we serialize to the map
        manager.register_pass<ov::pass::Serialize>(xml);
    }

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
    static std::mutex rtInfoMutex;

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

void IRSerializer::countModelSize() {
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

void IRSerializer::serializeModelToBuffer(uint8_t* xml, uint8_t* weights) {
    _logger.debug("serializeModelToBuffer");

    if (_serializeWeightsPtrToXml) {
        strcpy(reinterpret_cast<char*>(xml), _xmlString.c_str());
        xml[_xmlString.size()] = '\0';  // Null-terminate the string
        *weights = 0;                   // Placeholder for weights
    } else {
        writer_streambuf xmlStreamBuf(xml);
        writer_streambuf weightsStreamBuf(weights);
        std::ostream xmlStream(&xmlStreamBuf);
        std::ostream weightsStream(&weightsStreamBuf);

        serializeModelToStream(xmlStream, weightsStream);
    }

    _logger.debug("serializeModelToBuffer end");
}

}  // namespace intel_npu::driver_compiler_utils
