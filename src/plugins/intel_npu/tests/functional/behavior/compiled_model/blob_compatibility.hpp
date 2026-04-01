// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

const char* const BLOB_PREFIX = "blob_compat_";
const char* const OV_VERSION_PREFIX = "ov";
const char* const CIP_PREFIX = "cip";
const char* const BLOB_SUFFIX = ".blob";
const char* const BIN_SUFFIX = ".bin";

enum class E_DUMMY_MODELS { DUMMY_MODEL, DUMMY_MODEL_STATEFUL, DUMMY_MODEL_DYNAMIC_SHAPES };

const std::map<E_DUMMY_MODELS, std::string> DUMMY_MODELS{
    {E_DUMMY_MODELS::DUMMY_MODEL, "dummy_model"},
    {E_DUMMY_MODELS::DUMMY_MODEL_STATEFUL, "dummy_model_stateful"},
    {E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES, "dummy_model_dynamic_shapes"}};

enum class E_PLATFORMS { MTL, LNL, PTL };

const std::map<E_PLATFORMS, std::string> PLATFORMS{{E_PLATFORMS::MTL, "MTL"},
                                                   {E_PLATFORMS::LNL, "LNL"},
                                                   {E_PLATFORMS::PTL, "PTL"}};
const std::map<std::string, E_PLATFORMS> PARSED_PLATFORMS{{"3720", E_PLATFORMS::MTL},
                                                          {"4000", E_PLATFORMS::LNL},
                                                          {"5010", E_PLATFORMS::PTL}};

enum class E_OV_VERSIONS { OV_2025_4_0, OV_2026_0_0 };

const std::map<E_OV_VERSIONS, std::string> OV_VERSIONS{{E_OV_VERSIONS::OV_2025_4_0, "2025_4_0"},
                                                       {E_OV_VERSIONS::OV_2026_0_0, "2026_0_0"}};

enum class E_DRIVERS { DRIVER_1688, DRIVER_4511 };

const std::map<E_DRIVERS, std::string> DRIVERS{{E_DRIVERS::DRIVER_1688, "driver_1688"},
                                               {E_DRIVERS::DRIVER_4511, "driver_2020509"}};

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
        const auto& blobName = BLOB_PREFIX + model_name + "_" + platform + "_" + OV_VERSION_PREFIX + "_" + ov_release +
                               "_" + driver + BLOB_SUFFIX;
        blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + blobName;
        weightsPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + BLOB_PREFIX +
                      model_name + BIN_SUFFIX;
        config.emplace(ov::weights_path(weightsPath));
        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BlobCompatibilityParams>& obj) {
        std::string target_device, model_name, platform, ov_release, driver;
        ov::AnyMap config;
        std::tie(target_device, model_name, platform, ov_release, driver, config) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_blobName=\"" << BLOB_PREFIX << model_name << "_" << platform
               << "_" << OV_VERSION_PREFIX << "_" << ov_release << "_" << driver << BLOB_SUFFIX << "\"" << "_";

        if (!config.empty()) {
            for (auto& configItem : config) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    void importAndExportModel() {
        std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);
        ASSERT_TRUE(blobStream.is_open()) << "Failed to open blob file: " << blobPath;
        OV_ASSERT_NO_THROW(core.import_model(blobStream, target_device, config));

        std::shared_ptr<ov::Model> nullModel(nullptr);
        ov::CompiledModel compiledModel;
        config.emplace(ov::hint::compiled_blob(ov::read_tensor_data(blobPath)));
        OV_ASSERT_NO_THROW(compiledModel = core.compile_model(nullModel, target_device, config));
        config.erase(ov::hint::compiled_blob.name());

        std::ostringstream outBlobStream;
        OV_ASSERT_NO_THROW(compiledModel.export_model(outBlobStream));
        EXPECT_TRUE(outBlobStream.tellp() > 0);
    }

    ov::Core core;
    std::string blobPath;
    std::string weightsPath;
    ov::AnyMap config;
};

using compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw = OVBlobCompatibilityNPU;
using OVBlobCompatibilityNPU_Metadata_No_Throw = OVBlobCompatibilityNPU;
using OVBlobCompatibilityCiPNPU = OVBlobCompatibilityNPU;

TEST_P(OVBlobCompatibilityNPU, CanImportAllPrecompiledBlobsForAllOVVersionsAndDrivers) {
    importAndExportModel();
}

TEST_P(compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw, CanImportExpectedModelsForPVDriverAndAllOVVersions) {
    const char slashDelimiter = '/';
    const char backSlashDelimiter = '\\';
    const size_t lastSlashDelim = blobPath.find_last_of(slashDelimiter);
    const size_t lastBackSlashDelim = blobPath.find_last_of(backSlashDelimiter);

    size_t mtlPlatformPos = std::string::npos;
    if (lastSlashDelim != std::string::npos && lastBackSlashDelim != std::string::npos) {
        mtlPlatformPos = blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL), std::max(lastSlashDelim, lastBackSlashDelim));
    } else if (lastSlashDelim != std::string::npos) {
        mtlPlatformPos = blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL), lastSlashDelim);
    } else if (lastBackSlashDelim != std::string::npos) {
        mtlPlatformPos = blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL), lastBackSlashDelim);
    } else {
        mtlPlatformPos = blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL));
    }

    if (mtlPlatformPos == std::string::npos) {
        GTEST_SKIP() << "PV driver blob tests designed for NPU3720";
    }

    importAndExportModel();
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
