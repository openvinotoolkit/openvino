// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/version.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

// models generation
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"

namespace {

const char* const BLOB_PREFIX = "blob_compatibility_";
const char* const OV_VERSION_PREFIX = "ov";
const char* const DRIVER_PREFIX = "driver";
const char* const BLOB_SUFFIX = ".blob";

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

enum class E_OV_VERSIONS {
    OV_2024_6_0,
    OV_2025_0_0,
    OV_2025_1_0,
};

const std::map<E_OV_VERSIONS, std::string> OV_VERSIONS{{E_OV_VERSIONS::OV_2024_6_0, "2024_6_0"},
                                                       {E_OV_VERSIONS::OV_2025_0_0, "2025_0_0"},
                                                       {E_OV_VERSIONS::OV_2025_1_0, "2025_1_0"}};

enum class E_DRIVERS { DRIVER_1688, DRIVER_3967 };

const std::map<E_DRIVERS, std::string> DRIVERS{{E_DRIVERS::DRIVER_1688, "1688"}, {E_DRIVERS::DRIVER_3967, "1003967"}};

}  // namespace

namespace ov {
namespace test {
namespace behavior {

using BlobCompatibilityParams = std::tuple</* target_device = */ std::string,
                                           /* model_name = */ std::string,
                                           /* platform = */ std::string,
                                           /* ov_release = */ std::string,
                                           /* driver = */ std::string>;

class OVBlobCompatibilityNPU : public OVCompiledNetworkTestBase,
                               public testing::WithParamInterface<BlobCompatibilityParams> {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::string model_name, platform, ov_release, driver;
        std::tie(target_device, model_name, platform, ov_release, driver) = this->GetParam();
        blobName = BLOB_PREFIX + model_name + "_" + platform + "_" + OV_VERSION_PREFIX + "_" + ov_release + "_" +
                   DRIVER_PREFIX + "_" + driver + BLOB_SUFFIX;
        APIBaseTest::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<BlobCompatibilityParams> obj) {
        std::string target_device, model_name, platform, ov_release, driver;
        std::tie(target_device, model_name, platform, ov_release, driver) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_blobName=\"" << BLOB_PREFIX << model_name << "_" << platform
               << "_" << OV_VERSION_PREFIX << "_" << ov_release << "_" << DRIVER_PREFIX << "_" << driver << BLOB_SUFFIX
               << "\"";
        return result.str();
    }

protected:
    ov::Core core;
    std::string blobName;
};

using OVBlobCompatibilityNPU_PV_Driver_No_Throw = OVBlobCompatibilityNPU;

#define NO_APPEND_EXPORT(ASSERT_TYPE, ...)
#define APPEND_EXPORT(ASSERT_TYPE)                                                                           \
    std::shared_ptr<ov::Model> nullModel(nullptr);                                                           \
    ov::CompiledModel compiledModel;                                                                         \
    ASSERT_TYPE(compiledModel = core.compile_model(nullModel,                                                \
                                                   target_device,                                            \
                                                   {ov::hint::compiled_blob(ov::read_tensor_data(blobPath)), \
                                                    ov::intel_npu::disable_version_check(true)}));           \
    std::ostringstream outBlobStream;                                                                        \
    ASSERT_TYPE(compiledModel.export_model(outBlobStream));                                                  \
    EXPECT_TRUE(outBlobStream.tellp() > 0);

#define APPEND_EXPORT_HELPER_(arg1, arg2, arg3, ...) arg3
#define APPEND_EXPORT_HELPER(...)                    APPEND_EXPORT_HELPER_(__VA_ARGS__, NO_APPEND_EXPORT, APPEND_EXPORT)(__VA_ARGS__)

#define DEFAULT_TEST_BODY(ASSERT_TYPE, ...)                                                                    \
    const auto blobPath = ov::test::utils::NpuTestEnvConfig::getInstance().OV_NPU_TESTS_BLOBS_PATH + blobName; \
    std::ifstream blobStream(blobPath, std::ios::binary | std::ios::in);                                       \
    ASSERT_TYPE(core.import_model(blobStream, target_device, {ov::intel_npu::disable_version_check(true)}),    \
                ##__VA_ARGS__);                                                                                \
    APPEND_EXPORT_HELPER(ASSERT_TYPE, ##__VA_ARGS__)

TEST_P(OVBlobCompatibilityNPU, CanImportAllPrecompiledBlobsForAllOVVersionsAndDrivers) {
    if (auto current_driver =
            core.get_property(ov::test::utils::DEVICE_NPU, ov::intel_npu::driver_version.name()).as<std::string>();
        current_driver == DRIVERS.at(E_DRIVERS::DRIVER_1688) && blobName.find(current_driver) == std::string::npos) {
        GTEST_SKIP() << "FWD compatibility between drivers is not supported!";
    }
    DEFAULT_TEST_BODY(OV_ASSERT_NO_THROW);
}

TEST_P(OVBlobCompatibilityNPU_PV_Driver_No_Throw, CanImportExpectedModelsForPVDriverAndAllOVVersions) {
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
