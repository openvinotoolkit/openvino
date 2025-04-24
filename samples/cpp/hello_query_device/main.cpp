// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <iomanip>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"
// clang-format on

/**
 * @brief Print OV Parameters
 * @param reference on OV Parameter
 * @return void
 */
void print_any_value(const ov::Any& value) {
    if (value.empty()) {
        slog::info << "EMPTY VALUE" << slog::endl;
    } else {
        std::string stringValue = value.as<std::string>();
        slog::info << (stringValue.empty() ? "\"\"" : stringValue) << slog::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 1) {
            std::cout << "Usage : " << argv[0] << std::endl;
            return EXIT_FAILURE;
        }

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Get list of available devices --------
        std::vector<std::string> availableDevices = core.get_available_devices();

        // -------- Step 3. Query and print supported metrics and config keys --------
        slog::info << "Available devices: " << slog::endl;
        for (auto&& device : availableDevices) {
            slog::info << device << slog::endl;

            // Query supported properties and print all of them
            slog::info << "\tSUPPORTED_PROPERTIES: " << slog::endl;
            auto supported_properties = core.get_property(device, ov::supported_properties);
            for (auto&& property : supported_properties) {
                if (property != ov::supported_properties.name()) {
                    slog::info << "\t\t" << (property.is_mutable() ? "Mutable: " : "Immutable: ") << property << " : "
                               << slog::flush;
                    print_any_value(core.get_property(device, property));
                }
            }

            slog::info << slog::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
