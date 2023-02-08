// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <map>
#include <openvino/runtime/properties.hpp>
#include <string>
#include <threading/ie_istreams_executor.hpp>

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/property_supervisor.hpp"
#include "template/config.hpp"

namespace TemplatePlugin {

// ! [configuration:header]
struct RwProperties {
    RwProperties() {
        m_properties
            .add(
                ov::common_property(ov::device::id),
                [this] {
                    return std::to_string(deviceId);
                },
                [this](const std::string& str) {
                    deviceId = 0;
                },
                [](const std::string& str) {
                    OPENVINO_ASSERT(str == "0", "Unsupported device id: ", str);
                })
            .add(ov::common_property(ov::available_devices), {"0"})
            .add(ov::common_property(ov::enable_profiling), std::ref(enable_propfinling))
            .add(ov::template_plugin::throughput_streams)
            .add(/*ov::infer_property */ "INFERENCE", _streamsExecutorConfig.properties)
            .add(ov::inference_num_threads, std::ref(_streamsExecutorConfig._threads))
            .add(ov::hint::performance_mode, std::ref(performance_mode));
    }
    int deviceId = 0;
    bool enable_propfinling = true;
    InferenceEngine::IStreamsExecutor::Config _streamsExecutorConfig;
    ov::hint::PerformanceMode performance_mode = ov::hint::PerformanceMode::UNDEFINED;
    ov::PropertySupervisor m_properties;
};
// ! [configuration:header]

}  //  namespace TemplatePlugin
