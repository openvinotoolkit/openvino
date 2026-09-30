//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

int main(int argc, char* argv[]) {
    const std::string device = argc > 1 ? argv[1] : "NPU";
    try {
        ov::Core core;
        const auto offline_targets = core.get_property(device, ov::offline_compilation_targets);

#if 0
        std::cout << "Offline compilation targets for " << device << " (" << offline_targets.size() << "):" << std::endl;
        for (const auto& target : offline_targets) {
            std::cout << "  " << target << std::endl;
        }

        const auto selected_targets =
            core.get_property(device,
                              ov::offline_compilation_targets,
                              {{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)}});
        std::cout << "Offline compilation targets for " << device << " filtered by platform "
                  << ov::intel_npu::Platform::NPU6010 << " (" << selected_targets.size() << "):" << std::endl;
        for (const auto& target : selected_targets) {
            std::cout << "  " << target << std::endl;
        }
#endif
            std::string platform;
            std::cout << "Enter the target platform (e.g. 3720, 4000, 5010, 6010): ";
            if (!(std::cin >> platform)) {
                throw std::runtime_error("Failed to read NPU platform number");
            }

        // Check if target is available in the offline compilation targets
        const auto compile_target =
                std::find_if(offline_targets.begin(), offline_targets.end(), [&platform](const auto& target) {
                    return target.platform == platform;
            });
        if (compile_target == offline_targets.end()) {
                throw std::runtime_error("Platform " + platform + " is not present in the offline compilation targets");
        }

        // Create a simple model to compile
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        auto relu = std::make_shared<ov::op::v0::Relu>(input);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input});

        // Compile the model for the selected offline compilation target
        std::cout << "Compiling model for " << compile_target->platform << std::endl;
        auto compiled_model = core.compile_model(
            model,
            device,
            {ov::compilation_target(*compile_target)});
        std::cout << "Model compiled successfully for " << compile_target->platform << std::endl;

        // Exercise the export/import stub hook points
        std::stringstream blob;
        compiled_model.export_model(blob);
        std::cout << "Model exported successfully for " << compile_target->platform << std::endl;

        try {
            const auto imported_model = core.import_model(blob, device);
            std::cout << "Model imported successfully for " << compile_target->platform << std::endl;
        } catch (const std::exception& ex) {
            std::cout << "Model import failed (expected without a real device attached): " << ex.what() << std::endl;
        }
}
catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
}
return 0;
}
