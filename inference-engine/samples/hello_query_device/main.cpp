// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <ie_plugin_config.hpp>
#include <inference_engine.hpp>
#include <iomanip>
#include <memory>
#include <samples/common.hpp>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace InferenceEngine;

namespace {
/**
 * @brief Overload output stream operator to print vectors in pretty form
 * [value1, value2, ...]
 */
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v) {
    stream << "[ ";
    for (auto&& value : v)
        stream << value << " ";
    return stream << "]";
}

/**
 * @brief Print IE Parameters
 * @param reference on IE Parameter
 * @return void
 */
void printParameterValue(const Parameter& value) {
    if (value.empty()) {
        std::cout << "EMPTY VALUE" << std::endl;
    } else if (value.is<bool>()) {
        std::cout << std::boolalpha << value.as<bool>() << std::noboolalpha << std::endl;
    } else if (value.is<int>()) {
        std::cout << value.as<int>() << std::endl;
    } else if (value.is<unsigned int>()) {
        std::cout << value.as<unsigned int>() << std::endl;
    } else if (value.is<uint64_t>()) {
        std::cout << value.as<uint64_t>() << std::endl;
    } else if (value.is<float>()) {
        std::cout << value.as<float>() << std::endl;
    } else if (value.is<std::string>()) {
        std::string stringValue = value.as<std::string>();
        std::cout << (stringValue.empty() ? "\"\"" : stringValue) << std::endl;
    } else if (value.is<std::vector<std::string>>()) {
        std::cout << value.as<std::vector<std::string>>() << std::endl;
    } else if (value.is<std::vector<int>>()) {
        std::cout << value.as<std::vector<int>>() << std::endl;
    } else if (value.is<std::vector<float>>()) {
        std::cout << value.as<std::vector<float>>() << std::endl;
    } else if (value.is<std::vector<unsigned int>>()) {
        std::cout << value.as<std::vector<unsigned int>>() << std::endl;
    } else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        auto values = value.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
        std::cout << "{ ";
        std::cout << std::get<0>(values) << ", ";
        std::cout << std::get<1>(values) << ", ";
        std::cout << std::get<2>(values);
        std::cout << " }";
        std::cout << std::endl;
    } else if (value.is<Metrics::DeviceType>()) {
        auto v = value.as<Metrics::DeviceType>();
        std::cout << v << std::endl;
    } else if (value.is<std::map<InferenceEngine::Precision, float>>()) {
        auto values = value.as<std::map<InferenceEngine::Precision, float>>();
        std::cout << "{ ";
        for (auto& kv : values) {
            std::cout << kv.first << ": " << kv.second << "; ";
        }
        std::cout << " }";
        std::cout << std::endl;
    } else if (value.is<std::tuple<unsigned int, unsigned int>>()) {
        auto values = value.as<std::tuple<unsigned int, unsigned int>>();
        std::cout << "{ ";
        std::cout << std::get<0>(values) << ", ";
        std::cout << std::get<1>(values);
        std::cout << " }";
        std::cout << std::endl;
    } else {
        std::cout << "UNSUPPORTED TYPE" << std::endl;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (argc != 1) {
            std::cout << "Usage : " << argv[0] << std::endl;
            return EXIT_FAILURE;
        }

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        std::cout << "Loading Inference Engine" << std::endl;
        Core ie;

        // --------------------------- Get list of available devices
        // -------------------------------------

        std::vector<std::string> availableDevices = ie.GetAvailableDevices();

        // --------------------------- Query and print supported metrics and config
        // keys--------------------

        std::cout << "Available devices: " << std::endl;
        for (auto&& device : availableDevices) {
            std::cout << device << std::endl;

            std::cout << "\tSUPPORTED_METRICS: " << std::endl;
            std::vector<std::string> supportedMetrics = ie.GetMetric(device, METRIC_KEY(SUPPORTED_METRICS));
            for (auto&& metricName : supportedMetrics) {
                if (metricName != METRIC_KEY(SUPPORTED_METRICS) && metricName != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                    std::cout << "\t\t" << metricName << " : " << std::flush;
                    printParameterValue(ie.GetMetric(device, metricName));
                }
            }

            if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS)) != supportedMetrics.end()) {
                std::cout << "\tSUPPORTED_CONFIG_KEYS (default values): " << std::endl;
                std::vector<std::string> supportedConfigKeys = ie.GetMetric(device, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                for (auto&& configKey : supportedConfigKeys) {
                    std::cout << "\t\t" << configKey << " : " << std::flush;
                    printParameterValue(ie.GetConfig(device, configKey));
                }
            }

            std::cout << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
