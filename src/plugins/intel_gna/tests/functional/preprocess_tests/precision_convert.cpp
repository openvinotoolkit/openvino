// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov::test;

namespace LayerTestsDefinitions {

typedef std::tuple<ov::element::Type,                  // Net precision
                   ov::element::Type,                  // Input precision
                   ov::element::Type,                  // Output precision
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Configuration
                   >
    preprocessTestParamsSet;

class PreprocessGNATest : public testing::WithParamInterface<preprocessTestParamsSet>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<preprocessTestParamsSet>& obj) {
        ov::element::Type net_type, in_type, out_type;
        std::string target_device;
        std::map<std::string, std::string> conf;

        std::tie(net_type, in_type, out_type, target_device, conf) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << net_type << "_";
        result << "inPRC=" << in_type << "_";
        result << "outPRC=" << out_type << "_";
        result << "trgDev=" << target_device;
        for (auto const& conf_i : conf) {
            result << "_configItem=" << conf_i.first.c_str() << "_" << conf_i.second.c_str();
        }
        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type net_type;
        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({{8, 2}, {8, 2}});
        std::map<std::string, std::string> conf;
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();

        std::tie(net_type, inType, outType, targetDevice, conf) = this->GetParam();

        configuration.insert(conf.begin(), conf.end());

        init_input_shapes({input_shapes});

        ov::ParameterVector params;
        ov::OutputVector paramsOuts;
        for (auto&& shape : inputDynamicShapes) {
            auto param = std::make_shared<ov::op::v0::Parameter>(net_type, shape);
            params.push_back(param);
            paramsOuts.push_back(param);
        }
        auto concat = std::make_shared<ngraph::opset8::Concat>(paramsOuts, 1);
        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(concat)};
        function = std::make_shared<ngraph::Function>(results, params, "concat");
    }
};

class PreprocessGNAUnsupportedTest : public PreprocessGNATest {
public:
    PreprocessGNAUnsupportedTest(std::string error_str) : exp_error_str_(error_str) {}
    void run() override {
        try {
            PreprocessGNATest::compile_model();
            FAIL() << "GNA's unsupported layers were not detected during LoadNetwork()";
        } catch (std::runtime_error& e) {
            const std::string errorMsg = e.what();
            const auto expectedMsg = exp_error_str_;
            ASSERT_STR_CONTAINS(errorMsg, expectedMsg);
            EXPECT_TRUE(errorMsg.find(expectedMsg) != std::string::npos)
                << "Wrong error message, actual error message: " << errorMsg << ", expected: " << expectedMsg;
        }
    }

private:
    std::string exp_error_str_;
};

class PreprocessGNAUnsupportedInputsTest : public PreprocessGNAUnsupportedTest {
public:
    PreprocessGNAUnsupportedInputsTest()
        : PreprocessGNAUnsupportedTest("The plugin does not support input precision") {}
};

class PreprocessGNAUnsupportedOutputsTest : public PreprocessGNAUnsupportedTest {
public:
    PreprocessGNAUnsupportedOutputsTest() : PreprocessGNAUnsupportedTest("The plugin does not support layer") {}
};

TEST_P(PreprocessGNATest, CompareWithRefs) {
    run();
}

TEST_P(PreprocessGNAUnsupportedInputsTest, CompareWithRefs) {
    run();
}

TEST_P(PreprocessGNAUnsupportedOutputsTest, CompareWithRefs) {
    run();
}

std::map<std::string, std::string> config = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_SCALE_FACTOR_1", "1"}}};

ov::element::TypeVector inputTypesUnsupported = {ov::element::i32};

ov::element::TypeVector outputTypesUnsupported = {
    ov::element::u8,
    ov::element::i16,
};

ov::element::TypeVector outputTypesSupported = {ov::element::i32, ov::element::f32};

ov::element::TypeVector netTypes = {ov::element::f16, ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_Preprocess,
                         PreprocessGNATest,
                         ::testing::Combine(::testing::ValuesIn(netTypes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(outputTypesSupported),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(config)),
                         PreprocessGNATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Preprocess,
                         PreprocessGNAUnsupportedInputsTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes),
                                            ::testing::ValuesIn(inputTypesUnsupported),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(config)),
                         PreprocessGNATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Preprocess,
                         PreprocessGNAUnsupportedOutputsTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(outputTypesUnsupported),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(config)),
                         PreprocessGNATest::getTestCaseName);

}  // namespace LayerTestsDefinitions
