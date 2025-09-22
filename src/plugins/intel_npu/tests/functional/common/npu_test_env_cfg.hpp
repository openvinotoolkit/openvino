// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <openvino/runtime/device_id_parser.hpp>
#include <string>

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

namespace ov::test::utils {

/**
 * Reads configuration environment variables
 */
class NpuTestEnvConfig {
public:
    std::string IE_NPU_TESTS_DEVICE_NAME;
    std::string IE_NPU_TESTS_DUMP_PATH;
    std::string IE_NPU_TESTS_LOG_LEVEL;
    std::string IE_NPU_TESTS_PLATFORM;
    std::string OV_NPU_TESTS_SKIP_CONFIG_FILE = "npu_skip_func_tests.xml";
    mutable std::string OV_NPU_TESTS_BLOBS_PATH;  // mutable because it may have a default value set in main.cpp

    bool IE_NPU_TESTS_RUN_COMPILER = true;
    bool IE_NPU_TESTS_RUN_EXPORT = false;
    bool IE_NPU_TESTS_RUN_IMPORT = false;
    bool IE_NPU_TESTS_RUN_INFER = true;
    bool IE_NPU_TESTS_EXPORT_INPUT = false;
    bool IE_NPU_TESTS_EXPORT_OUTPUT = false;
    bool IE_NPU_TESTS_EXPORT_REF = false;
    bool IE_NPU_TESTS_IMPORT_INPUT = false;
    bool IE_NPU_TESTS_IMPORT_REF = false;

    bool IE_NPU_TESTS_RAW_EXPORT = false;
    bool IE_NPU_TESTS_LONG_FILE_NAME = false;

public:
    static const NpuTestEnvConfig& getInstance();

private:
    explicit NpuTestEnvConfig();
};

std::string getTestsDeviceNameFromEnvironmentOr(const std::string& instead);
std::string getTestsPlatformFromEnvironmentOr(const std::string& instead);

std::string getDeviceNameTestCase(const std::string& str);
std::string getDeviceName();
std::string getDeviceNameID(const std::string& str);

}  // namespace ov::test::utils

namespace InferRequestParamsAnyMapTestName {

std::string getTestCaseName(testing::TestParamInfo<ov::test::behavior::InferRequestParams> obj);

}  // namespace InferRequestParamsAnyMapTestName

namespace InferRequestParamsMapTestName {

typedef std::tuple<std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
    InferRequestParams;
std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj);

}  // namespace InferRequestParamsMapTestName
