// Copyright (C) 2018-2022 Intel Corporation
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
 * @brief Print IE Parameters
 * @param reference on IE Parameter
 * @return void
 */
void print_any_value(const ov::Any& value) {
    if (value.empty()) {
        slog::info << "EMPTY VALUE" << slog::endl;
    } else if (value.is<bool>()) {
        slog::info << std::boolalpha << value.as<bool>() << std::noboolalpha << slog::endl;
    } else if (value.is<int>()) {
        slog::info << value.as<int>() << slog::endl;
    } else if (value.is<unsigned int>()) {
        slog::info << value.as<unsigned int>() << slog::endl;
    } else if (value.is<uint64_t>()) {
        slog::info << value.as<uint64_t>() << slog::endl;
    } else if (value.is<float>()) {
        slog::info << value.as<float>() << slog::endl;
    } else if (value.is<std::string>()) {
        std::string stringValue = value.as<std::string>();
        slog::info << (stringValue.empty() ? "\"\"" : stringValue) << slog::endl;
    } else if (value.is<std::vector<std::string>>()) {
        slog::info << value.as<std::vector<std::string>>() << slog::endl;
    } else if (value.is<std::vector<int>>()) {
        slog::info << value.as<std::vector<int>>() << slog::endl;
    } else if (value.is<std::vector<float>>()) {
        slog::info << value.as<std::vector<float>>() << slog::endl;
    } else if (value.is<std::vector<unsigned int>>()) {
        slog::info << value.as<std::vector<unsigned int>>() << slog::endl;
    } else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        auto values = value.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
        slog::info << "{ ";
        slog::info << std::get<0>(values) << ", ";
        slog::info << std::get<1>(values) << ", ";
        slog::info << std::get<2>(values);
        slog::info << " }";
        slog::info << slog::endl;
    } else if (value.is<InferenceEngine::Metrics::DeviceType>()) {
        auto v = value.as<InferenceEngine::Metrics::DeviceType>();
        slog::info << v << slog::endl;
    } else if (value.is<std::map<InferenceEngine::Precision, float>>()) {
        auto values = value.as<std::map<InferenceEngine::Precision, float>>();
        slog::info << "{ ";
        for (auto& kv : values) {
            slog::info << kv.first << ": " << kv.second << "; ";
        }
        slog::info << " }";
        slog::info << slog::endl;
    } else if (value.is<std::tuple<unsigned int, unsigned int>>()) {
        auto values = value.as<std::tuple<unsigned int, unsigned int>>();
        slog::info << "{ ";
        slog::info << std::get<0>(values) << ", ";
        slog::info << std::get<1>(values);
        slog::info << " }";
        slog::info << slog::endl;
    } else {
        std::stringstream strm;
        value.print(strm);
        auto str = strm.str();
        if (str.empty()) {
            std::cout << "UNSUPPORTED TYPE" << std::endl;
        } else {
            std::cout << str << std::endl;
        }
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
        ov::runtime::Core core;

        // -------- Step 2. Get list of available devices --------
        std::vector<std::string> availableDevices = core.get_available_devices();

        // -------- Step 3. Query and print supported metrics and config keys --------
        slog::info << "Available devices: " << slog::endl;
        for (auto&& device : availableDevices) {
            slog::info << device << slog::endl;

            // Query supported metrics and print all of them
            slog::info << "\tSUPPORTED_METRICS: " << slog::endl;
            std::vector<std::string> supportedMetrics = core.get_metric(device, METRIC_KEY(SUPPORTED_METRICS));
            for (auto&& metricName : supportedMetrics) {
                if (metricName != METRIC_KEY(SUPPORTED_METRICS) && metricName != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                    slog::info << "\t\t" << metricName << " : " << slog::flush;
                    print_any_value(core.get_metric(device, metricName));
                }
            }

            // Query supported config keys and print all of them
            if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS)) !=
                supportedMetrics.end()) {
                slog::info << "\tSUPPORTED_CONFIG_KEYS (default values): " << slog::endl;
                std::vector<std::string> supportedConfigKeys =
                    core.get_metric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                for (auto&& configKey : supportedConfigKeys) {
                    slog::info << "\t\t" << configKey << " : " << slog::flush;
                    print_any_value(core.get_config(device, configKey));
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
