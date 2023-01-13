// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
namespace ov {
namespace device_utils {

bool is_device_supported(std::string name) {
    InferenceEngine::Core ie;
    auto devices = ie.GetAvailableDevices();
    for (auto&& device : devices)
        if (device.find(name) != std::string::npos) {
            return true;
        }
    return false;
}

bool is_gpu_device_supported() {
    return is_device_supported("GPU");
}

bool is_target_device_disabled(std::string device_name, std::string target_device) {
    if (!is_device_supported(device_name) && target_device.find(device_name) != std::string::npos)
        return true;
    else
        return false;
}

}  // namespace device_utils
}  // namespace ov
