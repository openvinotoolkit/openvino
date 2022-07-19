// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>
#include "openvino/runtime/intel_cpu/properties.hpp"

int main() {
    try {
        std::string modelPath = "modelWithDenormals.xml";
        std::string device = "CPU";
        ov::AnyMap config;
        //! [ov:intel_cpu:denormals_optimization:part0]
        ov::Core core;                                                    // Step 1: create ov::Core object
        core.set_property(ov::intel_cpu::denormals_optimization(true));   // Step 1b: Enable denormals optimization
        auto model = core.read_model(modelPath);                          // Step 2: Read Model
        //...                                                             // Step 3: Prepare inputs/outputs
        //...                                                             // Step 4: Set device configuration
        auto compiled = core.compile_model(model, device, config);        // Step 5: LoadNetwork
        //! [ov:intel_cpu:denormals_optimization:part0]
            if (!compiled) {
                throw std::runtime_error("error");
            }
    } catch (...) {
    }
    return 0;
}
