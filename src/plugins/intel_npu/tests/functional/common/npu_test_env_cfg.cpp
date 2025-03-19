// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"

#include <cstdlib>
#include <stdexcept>

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/config.hpp"

namespace ov::test::utils {

NpuTestEnvConfig::NpuTestEnvConfig() {
    // start reading obsolete environment variables
    if (auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        IE_NPU_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
        IE_NPU_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LOG_LEVEL")) {
        IE_NPU_TESTS_LOG_LEVEL = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_COMPILER")) {
        IE_NPU_TESTS_RUN_COMPILER = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_RUN_COMPILER", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_EXPORT")) {
        IE_NPU_TESTS_RUN_EXPORT = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_IMPORT")) {
        IE_NPU_TESTS_RUN_IMPORT = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_RUN_IMPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        IE_NPU_TESTS_RUN_INFER = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_RUN_INFER", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_INPUT")) {
        IE_NPU_TESTS_EXPORT_INPUT = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_OUTPUT")) {
        IE_NPU_TESTS_EXPORT_OUTPUT = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_REF")) {
        IE_NPU_TESTS_EXPORT_REF = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_INPUT")) {
        IE_NPU_TESTS_IMPORT_INPUT = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_REF")) {
        IE_NPU_TESTS_IMPORT_REF = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        IE_NPU_TESTS_RAW_EXPORT = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LONG_FILE_NAME")) {
        IE_NPU_TESTS_LONG_FILE_NAME = ::intel_npu::envVarStrToBool("IE_KMB_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_PLATFORM")) {
        IE_NPU_TESTS_PLATFORM = var;
    }
    // end reading obsolete environment variables

    if (auto var = std::getenv("IE_NPU_TESTS_DEVICE_NAME")) {
        IE_NPU_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_DUMP_PATH")) {
        IE_NPU_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_LOG_LEVEL")) {
        IE_NPU_TESTS_LOG_LEVEL = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_COMPILER")) {
        IE_NPU_TESTS_RUN_COMPILER = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_RUN_COMPILER", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_EXPORT")) {
        IE_NPU_TESTS_RUN_EXPORT = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_IMPORT")) {
        IE_NPU_TESTS_RUN_IMPORT = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_RUN_IMPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_INFER")) {
        IE_NPU_TESTS_RUN_INFER = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_RUN_INFER", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_INPUT")) {
        IE_NPU_TESTS_EXPORT_INPUT = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_OUTPUT")) {
        IE_NPU_TESTS_EXPORT_OUTPUT = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_REF")) {
        IE_NPU_TESTS_EXPORT_REF = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_IMPORT_INPUT")) {
        IE_NPU_TESTS_IMPORT_INPUT = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_IMPORT_REF")) {
        IE_NPU_TESTS_IMPORT_REF = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RAW_EXPORT")) {
        IE_NPU_TESTS_RAW_EXPORT = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_RAW_EXPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_LONG_FILE_NAME")) {
        IE_NPU_TESTS_LONG_FILE_NAME = ::intel_npu::envVarStrToBool("IE_NPU_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_PLATFORM")) {
        IE_NPU_TESTS_PLATFORM = var;
    }

    if (auto var = std::getenv("OV_NPU_TESTS_SKIP_CONFIG_FILE")) {
        OV_NPU_TESTS_SKIP_CONFIG_FILE = var;
    }
}

const NpuTestEnvConfig& NpuTestEnvConfig::getInstance() {
    static NpuTestEnvConfig instance{};
    return instance;
}

std::string getTestsDeviceNameFromEnvironmentOr(const std::string& instead) {
    return (!NpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME.empty())
               ? NpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME
               : instead;
}

std::string getTestsPlatformFromEnvironmentOr(const std::string& instead) {
    return (!NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM.empty())
               ? NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM
               : instead;
}

std::string getDeviceNameTestCase(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_name() + parser.get_device_id();
}

std::string getDeviceName() {
    return ov::test::utils::getTestsDeviceNameFromEnvironmentOr("NPU.3720");
}

std::string getDeviceNameID(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_id();
}

}  // namespace ov::test::utils

namespace InferRequestParamsAnyMapTestName {
std::string getTestCaseName(testing::TestParamInfo<ov::test::behavior::InferRequestParams> obj) {
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
            result << "_";
        }
    }
    return result.str();
}
}  // namespace InferRequestParamsAnyMapTestName

namespace InferRequestParamsMapTestName {

std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
    }
    return result.str();
}
}  // namespace InferRequestParamsMapTestName
