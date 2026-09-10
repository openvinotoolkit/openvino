// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <map>
#include <string>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
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

enum class E_PLATFORMS { MTL, LNL };

const std::map<E_PLATFORMS, std::string> PLATFORMS{{E_PLATFORMS::MTL, "MTL"}, {E_PLATFORMS::LNL, "LNL"}};
const std::map<std::string, E_PLATFORMS> PARSED_PLATFORMS{{"3720", E_PLATFORMS::MTL}, {"4000", E_PLATFORMS::LNL}};

enum class E_OV_VERSIONS { OV_2025_4_0 };

const std::map<E_OV_VERSIONS, std::string> OV_VERSIONS{{E_OV_VERSIONS::OV_2025_4_0, "2025_4_0"}};

enum class E_DRIVERS { DRIVER_1688, DRIVER_4511 };

const std::map<E_DRIVERS, std::string> DRIVERS{{E_DRIVERS::DRIVER_1688, "1688"}, {E_DRIVERS::DRIVER_4511, "2020509"}};

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
                               "_" + DRIVER_PREFIX + "_" + driver + BLOB_SUFFIX;
        blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + blobName;
        weightsPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + BLOB_PREFIX +
                      model_name + BIN_SUFFIX;
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
    std::string blobPath;
    std::string weightsPath;
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

#define DEFAULT_TEST_BODY(ASSERT_TYPE, ...)                                           \
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);              \
    config.emplace(ov::weights_path(weightsPath));                                    \
    ASSERT_TYPE(core.import_model(blobStream, target_device, config), ##__VA_ARGS__); \
    APPEND_EXPORT_HELPER(ASSERT_TYPE, ##__VA_ARGS__)

TEST_P(OVBlobCompatibilityNPU, CanImportAllPrecompiledBlobsForAllOVVersionsAndDrivers) {
    DEFAULT_TEST_BODY(OV_ASSERT_NO_THROW);
}

TEST_P(compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw, CanImportExpectedModelsForPVDriverAndAllOVVersions) {
    size_t mtlPlatformPos = std::string::npos;
    const char slashDelimiter = '/';
    const char backSlashDelimiter = '\\';
    size_t lastSlashDelim = blobPath.find_last_of(slashDelimiter);
    size_t lastBackSlashDelim = blobPath.find_last_of(backSlashDelimiter);
    if (lastSlashDelim != std::string::npos && lastBackSlashDelim != std::string::npos) {
        mtlPlatformPos = blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL), std::max(lastSlashDelim, lastBackSlashDelim));
    } else {
        mtlPlatformPos = blobPath.find(slashDelimiter) != std::string::npos
                             ? blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL), blobPath.rfind(slashDelimiter))
                         : blobPath.find(backSlashDelimiter) != std::string::npos
                             ? blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL), blobPath.rfind(backSlashDelimiter))
                             : blobPath.find(PLATFORMS.at(E_PLATFORMS::MTL));
    }
    if (mtlPlatformPos == std::string::npos) {
        GTEST_SKIP() << "PV driver blob tests designed for NPU3720";
    }
    DEFAULT_TEST_BODY(OV_ASSERT_NO_THROW);
}

#undef NO_APPEND_EXPORT
#undef APPEND_EXPORT
#undef APPEND_EXPORT_HELPER_
#undef APPEND_EXPORT_HELPER
#undef DEFAULT_TEST_BODY

}  // namespace behavior

}  // namespace test

}  // namespace ov

using namespace ov::test::behavior;

const auto all_models = []() -> std::vector<std::string> {
    std::vector<std::string> models(DUMMY_MODELS.size());
    std::transform(DUMMY_MODELS.begin(),
                   DUMMY_MODELS.end(),
                   models.begin(),
                   [](const decltype(DUMMY_MODELS)::value_type& pair) {
                       return pair.second;
                   });
    return models;
}();

const auto match_platform =
    !ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM.empty()
        ? (PARSED_PLATFORMS.find(ov::intel_npu::platforms::standardize(
               ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM)) != PARSED_PLATFORMS.end()
               ? PLATFORMS.at(PARSED_PLATFORMS.at(ov::intel_npu::platforms::standardize(
                     ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM)))
               : "")
        : "";

const auto all_ov_releases = []() -> std::vector<std::string> {
    std::vector<std::string> ov_releases(OV_VERSIONS.size());
    std::transform(OV_VERSIONS.begin(),
                   OV_VERSIONS.end(),
                   ov_releases.begin(),
                   [](const decltype(OV_VERSIONS)::value_type& pair) {
                       return pair.second;
                   });
    return ov_releases;
}();

const auto all_drivers = []() -> std::vector<std::string> {
    std::vector<std::string> drivers(DRIVERS.size());
    std::transform(DRIVERS.begin(), DRIVERS.end(), drivers.begin(), [](const decltype(DRIVERS)::value_type& pair) {
        return pair.second;
    });
    return drivers;
}();

const auto pv_compatible_models = []() -> std::vector<std::string> {
    std::vector<std::string> models(all_models.size() - 1);
    std::copy_if(all_models.begin(), all_models.end(), models.begin(), [](const std::string& model) {
        return DUMMY_MODELS.at(E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES) != model;
    });
    return models;
}();

const auto all_drivers_except_pv = []() -> std::vector<std::string> {
    std::vector<std::string> drivers(all_drivers.size() - 1);
    std::copy_if(all_drivers.begin(), all_drivers.end(), drivers.begin(), [](const std::string& driver) {
        return DRIVERS.at(E_DRIVERS::DRIVER_1688) != driver;
    });
    return drivers;
}();

const std::vector<ov::AnyMap> config = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(all_models),
                                            ::testing::Values(match_platform),
                                            ::testing::ValuesIn(all_ov_releases),
                                            ::testing::ValuesIn(all_drivers_except_pv),
                                            ::testing::ValuesIn(config)),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>);

#ifdef _WIN32  // Linux supports only ELF backend
INSTANTIATE_TEST_SUITE_P(
    compatibility_smoke_Behavior_NPU,
    compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(pv_compatible_models),
                       ::testing::Values(match_platform),
                       ::testing::ValuesIn(all_ov_releases),
                       ::testing::Values(DRIVERS.at(E_DRIVERS::DRIVER_1688)),
                       ::testing::ValuesIn(config)),
    ov::test::utils::appendPlatformTypeTestName<compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw>);
#endif
