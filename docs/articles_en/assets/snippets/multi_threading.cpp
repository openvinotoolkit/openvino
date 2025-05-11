// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>

#include "openvino/runtime/intel_cpu/properties.hpp"

int main() {
    try {
        std::string modelPath = "model.xml";
        std::string device = "CPU";
        ov::AnyMap config;
        ov::Core core;
        core.set_property(ov::inference_num_threads(1));
        auto model = core.read_model(modelPath);
        //! [ov:intel_cpu:multi_threading:part0]
        // Use one logical processor for inference
        auto compiled_model_1 = core.compile_model(model, device, ov::inference_num_threads(1));

        // Use logical processors of Efficient-cores for inference on hybrid platform
        auto compiled_model_2 = core.compile_model(model, device, ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY));

        // Use one logical processor per CPU core for inference when hyper threading is on
        auto compiled_model_3 = core.compile_model(model, device, ov::hint::enable_hyper_threading(false));
        //! [ov:intel_cpu:multi_threading:part0]

        //! [ov:intel_cpu:multi_threading:part1]
        // Disable CPU threads pinning for inference when system support it
        auto compiled_model_4 = core.compile_model(model, device, ov::hint::enable_cpu_pinning(false));
        //! [ov:intel_cpu:multi_threading:part1]
        if ((!compiled_model_1) || (!compiled_model_2) || (!compiled_model_3) || (!compiled_model_4)) {
            throw std::runtime_error("error");
        }
    } catch (...) {
    }
    return 0;
}
