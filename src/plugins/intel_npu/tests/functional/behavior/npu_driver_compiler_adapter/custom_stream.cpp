// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include <random>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "graph_transformations.hpp"
#include "intel_npu/al/config/common.hpp"
#include "openvino/opsets/opset11.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using IR = intel_npu::driverCompilerAdapter::IR;

namespace ov::test::behavior {

class DriverCompilerAdapterCustomStreamTestNPU : public ov::test::behavior::OVPluginTestBase,
                                                 public testing::WithParamInterface<CompilationParams> {
public:
    std::shared_ptr<ov::Model> createModelWithLargeSize() {
        auto data = std::make_shared<ov::opset11::Parameter>(ov::element::f16, ov::Shape{4000, 4000});
        auto mul_constant = ov::opset11::Constant::create(ov::element::f16, ov::Shape{1}, {1.5});
        auto mul = std::make_shared<ov::opset11::Multiply>(data, mul_constant);
        auto add_constant = ov::opset11::Constant::create(ov::element::f16, ov::Shape{1}, {0.5});
        auto add = std::make_shared<ov::opset11::Add>(mul, add_constant);
        // Just a sample model here, large iteration to make the model large
        for (int i = 0; i < 1000; i++) {
            add = std::make_shared<ov::opset11::Add>(add, add_constant);
        }
        auto res = std::make_shared<ov::opset11::Result>(add);

        /// Create the OpenVINO model
        return std::make_shared<ov::Model>(ov::ResultVector{std::move(res)}, ov::ParameterVector{std::move(data)});
    }

    std::string generateRandomFileName() {
        std::stringstream ss;
        auto now = std::chrono::high_resolution_clock::now();
        auto seed = now.time_since_epoch().count();
        std::mt19937 mt_rand(seed);
        std::uniform_int_distribution<int> dist(0, 15);

        for (unsigned int i = 0; i < 16; ++i) {
            int random_number = dist(mt_rand);
            ss << std::hex << random_number;
        }
        return ss.str();
    }

    size_t getFileSize(std::istream& strm) {
        const size_t streamStart = strm.tellg();
        strm.seekg(0, std::ios_base::end);
        const size_t streamEnd = strm.tellg();
        const size_t bytesAvailable = streamEnd - streamStart;
        strm.seekg(streamStart, std::ios_base::beg);
        return bytesAvailable;
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
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
        APIBaseTest::TearDown();
    }

protected:
    ov::AnyMap configuration;
};

TEST_P(DriverCompilerAdapterCustomStreamTestNPU, TestLargeModel) {
    auto model = createModelWithLargeSize();
    IR irModel(model, 11, true);
    std::istream& xmlStream = irModel.getXml();
    std::istream& weightsStream = irModel.getWeights();
    size_t xmlSize = getFileSize(xmlStream);
    size_t weightsSize = getFileSize(weightsStream);
    std::string fileName = generateRandomFileName();
    std::string xmlFileName = fileName + ".xml";
    std::string binFileName = fileName + ".bin";
    {
        std::vector<char> xml(xmlSize);
        xmlStream.read(xml.data(), xmlSize);
        std::ofstream xmlFile(xmlFileName, std::ios::binary);
        if (xmlFile) {
            xmlFile.write(xml.data(), xmlSize);
            xmlFile.close();
        }

        std::vector<char> weights(weightsSize);
        weightsStream.read(weights.data(), weightsSize);
        std::ofstream binFile(binFileName, std::ios::binary);
        if (binFile) {
            binFile.write(weights.data(), weightsSize);
            binFile.close();
        }
    }
    ov::Core core;
    EXPECT_NO_THROW(model = core.read_model(xmlFileName));
    if (std::remove(xmlFileName) != 0 || std::remove(binFileName) != 0) {
        OPENVINO_THROW("Failed to remove serialized files");
    }
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