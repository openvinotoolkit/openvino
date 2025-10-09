// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include <random>

#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/opsets/opset11.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "vcl_serializer.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using VCLSerializerWithWeightsCopy = intel_npu::driver_compiler_utils::VCLSerializerWithWeightsCopy;

namespace ov::test::behavior {

class DriverCompilerAdapterCustomStreamTestNPU : public ov::test::behavior::OVPluginTestBase,
                                                 public testing::WithParamInterface<CompilationParams> {
public:
    std::string generateRandomFileName() {
        std::stringstream ss;
        auto now = std::chrono::high_resolution_clock::now();
        auto seed = now.time_since_epoch().count();
        std::mt19937 mt_rand(static_cast<unsigned int>(seed));
        std::uniform_int_distribution<int> dist(0, 15);

        for (unsigned int i = 0; i < 16; ++i) {
            int random_number = dist(mt_rand);
            ss << std::hex << random_number;
        }
        return ss.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        std::string fileName = generateRandomFileName();
        xmlFileName = fileName + ".xml";
        binFileName = fileName + ".bin";
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        if (std::remove(xmlFileName.c_str()) != 0 || std::remove(binFileName.c_str()) != 0) {
            ADD_FAILURE() << "Failed to remove serialized files, xml: " << xmlFileName << " bin: " << binFileName;
        }
        APIBaseTest::TearDown();
    }

protected:
    ov::AnyMap configuration;
    std::string xmlFileName;
    std::string binFileName;
};

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, TestLargeModel) {
    auto model = createModelWithLargeSize();
    const ze_graph_compiler_version_info_t dummyCompilerVersion{0, 0};
    VCLSerializerWithWeightsCopy VCLSerializerWithWeightsCopy(model, dummyCompilerVersion, 11);
    EXPECT_NO_THROW(VCLSerializerWithWeightsCopy.serialize());
}

const std::vector<ov::AnyMap> configs = {
    {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DriverCompilerAdapterCustomStreamTestNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         DriverCompilerAdapterCustomStreamTestNPU::getTestCaseName);
}  // namespace ov::test::behavior
