// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_serializer.hpp"

#include <cstdint>
#include <istream>
#include <mutex>
#include <streambuf>

#include "openvino/pass/serialize.hpp"

namespace intel_npu::driver_compiler_utils {

IRSerializer::IRSerializer(const std::shared_ptr<const ov::Model>& origModel)
    : _logger("IRSerializer", Logger::global().level()) {
    // There is no const variant of run_passes so use const_cast here
    // as model serialization does not mutate the model
    _model = std::const_pointer_cast<ov::Model>(origModel);

    countModelSize();
}

void IRSerializer::serializeModelToStream(std::ostream& xml, std::ostream& weights) {
    _logger.debug("serializeModelToStream");
    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(passConfig, "NPU:serializeModelToStream");
    manager.register_pass<ov::pass::Serialize>(xml, weights);

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    const auto newAPIKey = "is_new_api";

    // Flag used for indicating an NPU plugin version which switched the I/O identification convention from names to
    // indices. The flag is required in order to inform the driver-compiler adapter to expect indices when attempting to
    // deserialize the I/O metadata.
    const auto useIndicesForIOMetadataKey = "use_indices_for_io_metadata";

    // We modify the original model object here therefore a mutex is required
    static std::mutex rtInfoMutex;

    {
        std::lock_guard<std::mutex> lock(rtInfoMutex);

        _model->set_rt_info(true, newAPIKey);
        _model->set_rt_info(true, useIndicesForIOMetadataKey);

        manager.run_passes(_model);

        auto& rtInfo = _model->get_rt_info();
        rtInfo.erase(newAPIKey);
        rtInfo.erase(useIndicesForIOMetadataKey);
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

    writer_streambuf xmlStreamBuf(xml);
    writer_streambuf weightsStreamBuf(weights);
    std::ostream xmlStream(&xmlStreamBuf);
    std::ostream weightsStream(&weightsStreamBuf);

    serializeModelToStream(xmlStream, weightsStream);

    _logger.debug("serializeModelToBuffer end");
}

}  // namespace intel_npu::driver_compiler_utils
