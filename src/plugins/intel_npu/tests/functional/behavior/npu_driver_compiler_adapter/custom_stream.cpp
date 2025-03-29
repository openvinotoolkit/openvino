// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include <random>

#include "base/ov_behavior_test_utils.hpp"
#include "common/functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "intel_npu/config/common.hpp"
#include "ir_serializer.hpp"
#include "openvino/opsets/opset11.hpp"


using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using IRSerializer = intel_npu::driver_compiler_utils::IRSerializer;

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

    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
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
    IRSerializer irSerializer(model, 11);
    size_t xmlSize = irSerializer.getXmlSize();
    size_t weightsSize = irSerializer.getWeightsSize();

    std::vector<uint8_t> xml(xmlSize);
    std::vector<uint8_t> weights(weightsSize);
    irSerializer.serializeModelToBuffer(xml.data(), weights.data());

    {
        std::ofstream xmlFile(xmlFileName, std::ios::binary);
        if (xmlFile) {
            xmlFile.write(reinterpret_cast<const char*>(xml.data()), xmlSize);
            xmlFile.close();
        }

        std::ofstream binFile(binFileName, std::ios::binary);
        if (binFile) {
            binFile.write(reinterpret_cast<const char*>(weights.data()), weightsSize);
            binFile.close();
        }
    }
    ov::Core core;
    EXPECT_NO_THROW(model = core.read_model(xmlFileName));
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
