// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

const char* const BLOB_PREFIX = "blob_compat_";
const char* const OV_VERSION_PREFIX = "ov";
const char* const DRIVER_PREFIX = "driver";
const char* const BLOB_SUFFIX = ".blob";
const char* const BIN_SUFFIX = ".bin";

enum class E_DUMMY_MODELS { DUMMY_MODEL, DUMMY_MODEL_STATEFUL, DUMMY_MODEL_DYNAMIC_SHAPES };

const std::map<E_DUMMY_MODELS, std::string> DUMMY_MODELS{
    {E_DUMMY_MODELS::DUMMY_MODEL, "dummy_model"},
    {E_DUMMY_MODELS::DUMMY_MODEL_STATEFUL, "dummy_model_stateful"},
    {E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES, "dummy_model_dynamic_shapes"}};

enum class E_PLATFORMS {
    MTL,
};

const std::map<E_PLATFORMS, std::string> PLATFORMS{{E_PLATFORMS::MTL, "MTL"}};
const std::map<std::string, E_PLATFORMS> PARSED_PLATFORMS{{"NPU3720", E_PLATFORMS::MTL}};

enum class E_OV_VERSIONS { OV_2024_6_0, OV_2025_0_0, OV_2025_1_0, OV_2025_3_0 };

const std::map<E_OV_VERSIONS, std::string> OV_VERSIONS{{E_OV_VERSIONS::OV_2024_6_0, "2024_6_0"},
                                                       {E_OV_VERSIONS::OV_2025_0_0, "2025_0_0"},
                                                       {E_OV_VERSIONS::OV_2025_1_0, "2025_1_0"},
                                                       {E_OV_VERSIONS::OV_2025_3_0, "2025_3_0"}};

enum class E_DRIVERS { DRIVER_1688, DRIVER_3967, DRIVER_4297 };

const std::map<E_DRIVERS, std::string> DRIVERS{{E_DRIVERS::DRIVER_1688, "1688"},
                                               {E_DRIVERS::DRIVER_3967, "1003967"},
                                               {E_DRIVERS::DRIVER_4297, "2020426"}};

}  // namespace

namespace ov {
namespace test {
namespace behavior {

using BlobCompatibilityParams = std::tuple</* target_device = */ std::string,
                                           /* model_name = */ std::string,
                                           /* platform = */ std::string,
                                           /* ov_release = */ std::string,
                                           /* driver = */ std::string,
                                           /* config = */ ov::AnyMap>;

class OVBlobCompatibilityNPU : public OVCompiledNetworkTestBase,
                               public testing::WithParamInterface<BlobCompatibilityParams> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::string model_name, platform, ov_release, driver;
        std::tie(target_device, model_name, platform, ov_release, driver, config) = this->GetParam();
        blobName = BLOB_PREFIX + model_name + "_" + platform + "_" + OV_VERSION_PREFIX + "_" + ov_release + "_" +
                   DRIVER_PREFIX + "_" + driver + BLOB_SUFFIX;
        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BlobCompatibilityParams>& obj) {
        std::string target_device, model_name, platform, ov_release, driver;
        ov::AnyMap config;
        std::tie(target_device, model_name, platform, ov_release, driver, config) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_blobName=\"" << BLOB_PREFIX << model_name << "_" << platform
               << "_" << OV_VERSION_PREFIX << "_" << ov_release << "_" << DRIVER_PREFIX << "_" << driver << BLOB_SUFFIX
               << "\"" << "_";

        if (!config.empty()) {
            for (auto& configItem : config) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::Core core;
    std::string blobName;
    ov::AnyMap config;
};

using compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw = OVBlobCompatibilityNPU;
using OVBlobCompatibilityNPU_Metadata_No_Throw = OVBlobCompatibilityNPU;

#define NO_APPEND_EXPORT(ASSERT_TYPE, ...)
#define APPEND_EXPORT(ASSERT_TYPE)                                                     \
    std::shared_ptr<ov::Model> nullModel(nullptr);                                     \
    ov::CompiledModel compiledModel;                                                   \
    config.emplace(ov::hint::compiled_blob(ov::read_tensor_data(blobPath)));           \
    ASSERT_TYPE(compiledModel = core.compile_model(nullModel, target_device, config)); \
    config.erase(ov::hint::compiled_blob.name());                                      \
    std::ostringstream outBlobStream;                                                  \
    ASSERT_TYPE(compiledModel.export_model(outBlobStream));                            \
    EXPECT_TRUE(outBlobStream.tellp() > 0);

#define APPEND_EXPORT_HELPER_(arg1, arg2, arg3, ...) arg3
#define APPEND_EXPORT_HELPER(...)                    APPEND_EXPORT_HELPER_(__VA_ARGS__, NO_APPEND_EXPORT, APPEND_EXPORT)(__VA_ARGS__)

#define DEFAULT_TEST_BODY(ASSERT_TYPE, ...)                                                                    \
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + blobName; \
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);                                       \
    ASSERT_TYPE(core.import_model(blobStream, target_device, config), ##__VA_ARGS__);                          \
    APPEND_EXPORT_HELPER(ASSERT_TYPE, ##__VA_ARGS__)

TEST_P(OVBlobCompatibilityNPU, CanImportAllPrecompiledBlobsForAllOVVersionsAndDrivers) {
    DEFAULT_TEST_BODY(OV_ASSERT_NO_THROW);
}

TEST_P(compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw, CanImportExpectedModelsForPVDriverAndAllOVVersions) {
    DEFAULT_TEST_BODY(OV_ASSERT_NO_THROW);
}

TEST_P(OVBlobCompatibilityNPU_Metadata_No_Throw, CanImportOlderMetadata) {
    std::string blobWeightsName(blobName.data(), blobName.find(BLOB_SUFFIX));
    blobWeightsName += BIN_SUFFIX;
    config.insert(
        ov::weights_path(ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + blobWeightsName));
    DEFAULT_TEST_BODY(OV_ASSERT_NO_THROW);
    config.erase(ov::weights_path.name());
}

#undef NO_APPEND_EXPORT
#undef APPEND_EXPORT
#undef APPEND_EXPORT_HELPER_
#undef APPEND_EXPORT_HELPER
#undef DEFAULT_TEST_BODY

}  // namespace behavior

}  // namespace test

}  // namespace ov
