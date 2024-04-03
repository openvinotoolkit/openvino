// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformations.hpp"

#include <istream>
#include <mutex>

#include "openvino/pass/serialize.hpp"
#include "transformations/op_conversions/convert_interpolate11_downgrade.hpp"

namespace intel_npu::driverCompilerAdapter {

IR serializeToIR(const std::shared_ptr<const ov::Model>& origModel, uint32_t supportedOpset) {
    // There is no const variant of run_passes so use const_cast here
    // as model serialization does not mutate the model
    std::shared_ptr<ov::Model> model = std::const_pointer_cast<ov::Model>(origModel);

    const auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager(passConfig);

    if (supportedOpset < 11) {
        // Need to clone to modify the model and remain thread safe
        model = model->clone();
        // Downgrade to opset10
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }

    std::stringstream xmlStream, weightsStream;
    manager.register_pass<ov::pass::Serialize>(xmlStream, weightsStream);

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    const auto new_api_key = "is_new_api";

    // We modify the original model object here therefore a mutex is required
    static std::mutex rtInfoMutex;

    {
        std::lock_guard<std::mutex> lock(rtInfoMutex);

        model->set_rt_info(true, new_api_key);

        manager.run_passes(model);

        auto& rtInfo = model->get_rt_info();
        rtInfo.erase(new_api_key);
    }

    return {std::move(xmlStream), std::move(weightsStream)};
}

}  // namespace intel_npu::driverCompilerAdapter
