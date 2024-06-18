// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformations.hpp"

#include <istream>
#include <mutex>
#include <streambuf>

#include "openvino/pass/serialize.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

namespace intel_npu::driverCompilerAdapter {

IR::IR(const std::shared_ptr<const ov::Model>& origModel, uint32_t supportedOpset, bool largeModel)
    : _logger("LevelZeroCompilerAdapter::IR", Logger::global().level()),
      _isLargeModel(largeModel),
      _xmlStream(&_xmlCache),
      _weightsStream(&_weightsCache) {
    // There is no const variant of run_passes so use const_cast here
    // as model serialization does not mutate the model
    _model = std::const_pointer_cast<ov::Model>(origModel);

#ifdef _WIN32
    // Only use fstream for Windows
    if (_model->get_graph_size() > 1024U * 1024 * 1024 * 2) {
        // Force model larger than 2G to use FILE mode
        _logger.warning("Force large model %s to use custom stream to do serialize", _model->get_friendly_name());
        _isLargeModel = true;
    }
#endif

    serializeToIR(supportedOpset);
}

IR::~IR() {}

void IR::serializeToIR(uint32_t supportedOpset) {
    _logger.debug("serializeToIR");
    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(passConfig);

    if (supportedOpset < 11) {
        // Need to clone to modify the model and remain thread safe
        _model = _model->clone();
        // Downgrade to opset10
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }

    if (_isLargeModel) {
        manager.register_pass<ov::pass::Serialize>(_xmlStream, _weightsStream);
        _logger.info("Serialize to custom stream");
    } else {
        manager.register_pass<ov::pass::Serialize>(_xml, _weights);
        _logger.info("Serialize to normal stream");
    }

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    const auto new_api_key = "is_new_api";

    // We modify the original model object here therefore a mutex is required
    static std::mutex rtInfoMutex;

    {
        std::lock_guard<std::mutex> lock(rtInfoMutex);

        _model->set_rt_info(true, new_api_key);

        manager.run_passes(_model);

        auto& rtInfo = _model->get_rt_info();
        rtInfo.erase(new_api_key);
    }

    _logger.debug("serializeToIR end");
}

}  // namespace intel_npu::driverCompilerAdapter
