// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <map>
#include <openvino/runtime/properties.hpp>
#include <string>
#include <template/properties.hpp>
#include <threading/ie_istreams_executor.hpp>

#include "properties.hpp"

namespace TemplatePlugin {

// ! [configuration:header]
struct RwProperties {
    RwProperties() {
        _properties
            .add(
                ov::device::id,
                [this] {
                    return std::to_string(deviceId);
                },
                [this](const std::string& str) {
                    deviceId = 0;
                },
                [](const std::string& str) {
                    OPENVINO_ASSERT(str == "0", "Unsupported device id: ", str);
                })
            .add(ov::available_devices, {"0"})
            .add(ov::template_plugin::ro_property)
            .add(ov::template_plugin::rw_property, ov::template_plugin::Value::UNDEFINED)
            .add(ov::enable_profiling, std::ref(enable_propfinling))
            .add(ov::infer_property, _streamsExecutorConfig.properties)
            .add(ov::hint::performance_mode, std::ref(performance_mode));
    }
    int deviceId = 0;
    bool enable_propfinling = true;
    InferenceEngine::IStreamsExecutor::Config _streamsExecutorConfig;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::UNDEFINED;
    ov::PropertyAccess _properties;
};
// ! [configuration:header]

}  //  namespace TemplatePlugin
