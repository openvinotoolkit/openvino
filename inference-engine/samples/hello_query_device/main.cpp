// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iomanip>
#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <cstdlib>

#include <samples/common.hpp>

#include <inference_engine.hpp>

using namespace InferenceEngine;

namespace {

template <typename T>
std::ostream & operator << (std::ostream & stream, const std::vector<T> & v) {
    stream << "[ ";
    for (auto && value : v)
        stream << value << " ";
    return stream << "]";
}

void printParameterValue(const Parameter & value) {
    if (value.is<bool>()) {
        std::cout << std::boolalpha << value.as<bool>() << std::noboolalpha << std::endl;
    } else if (value.is<int>()) {
        std::cout << value.as<int>() << std::endl;
    } else if (value.is<unsigned int>()) {
        std::cout << value.as<unsigned int>() << std::endl;
    } else if (value.is<float>()) {
        std::cout << value.as<float>() << std::endl;
    } else if (value.is<std::string>()) {
        std::string stringValue = value.as<std::string>();
        std::cout << (stringValue.empty() ? "\"\"" : stringValue) << std::endl;
    } else if (value.is<std::vector<std::string> >()) {
        std::cout << value.as<std::vector<std::string> >() << std::endl;
    } else if (value.is<std::vector<int> >()) {
        std::cout << value.as<std::vector<int> >() << std::endl;
    } else if (value.is<std::vector<float> >()) {
        std::cout << value.as<std::vector<float> >() << std::endl;
    } else if (value.is<std::vector<unsigned int> >()) {
        std::cout << value.as<std::vector<unsigned int> >() << std::endl;
    } else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int> >()) {
        auto values = value.as<std::tuple<unsigned int, unsigned int, unsigned int> >();
        std::cout << "{ ";
        std::cout << std::get<0>(values) << ", ";
        std::cout << std::get<1>(values) << ", ";
        std::cout << std::get<2>(values);
        std::cout << " }";
        std::cout << std::endl;
    } else if (value.is<std::tuple<unsigned int, unsigned int> >()) {
        auto values = value.as<std::tuple<unsigned int, unsigned int> >();
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

int main(int argc, char *argv[]) {
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 1) {
            std::cout << "Usage : ./hello_query_device" << std::endl;
            return EXIT_FAILURE;
        }

        // --------------------------- 1. Load Inference engine instance -------------------------------------

        Core ie;

        // --------------------------- 2. Get list of available devices  -------------------------------------

        std::vector<std::string> availableDevices = ie.GetAvailableDevices();

        // --------------------------- 3. Query and print supported metrics and config keys--------------------
        std::set<std::string> printedDevices;

        std::cout << "Available devices: " << std::endl;
        for (auto && device : availableDevices) {
            std::string deviceFamilyName = device.substr(0, device.find_first_of('.'));
            if (printedDevices.find(deviceFamilyName) == printedDevices.end())
                printedDevices.insert(deviceFamilyName);
            else
                continue;

            std::cout << "\tDevice: " << deviceFamilyName << std::endl;

            std::cout << "\tMetrics: " << std::endl;
            std::vector<std::string> supportedMetrics = ie.GetMetric(deviceFamilyName, METRIC_KEY(SUPPORTED_METRICS));
            for (auto && metricName : supportedMetrics) {
                std::cout << "\t\t" << metricName << " : " << std::flush;
                printParameterValue(ie.GetMetric(device, metricName));
            }

            std::cout << "\tDefault values for device configuration keys: " << std::endl;
            std::vector<std::string> supportedConfigKeys = ie.GetMetric(deviceFamilyName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            for (auto && configKey : supportedConfigKeys) {
                std::cout << "\t\t" << configKey << " : " << std::flush;
                printParameterValue(ie.GetConfig(deviceFamilyName, configKey));
            }

            std::cout << std::endl;
        }
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
