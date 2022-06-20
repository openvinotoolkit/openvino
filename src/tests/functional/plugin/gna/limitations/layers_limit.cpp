// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/opsets/opset8.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace InferenceEngine;
// using namespace ov::test;

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::string,            // Device name
        std::map<std::string, std::string>, //common config
        std::map<std::string, std::string>,  // Configuration
        size_t // number of layers
> GNALayersLimitTestParams;

class GNALayersLimitTest : public testing::WithParamInterface<GNALayersLimitTestParams>,
                           public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GNALayersLimitTestParams>& obj) {
        std::string target_device;
        std::map<std::string, std::string> common_conf, conf;
        size_t layers_number;

        std::tie(target_device, common_conf, conf, layers_number) = obj.param;

        std::ostringstream result;
        result << "layerNumber=" << layers_number;
        result << "_trgDev=" << target_device;
        for (auto const &conf_i : conf) {
            result << "_configItem=" << conf_i.first.c_str() << "_" << conf_i.second.c_str();
        }
        return result.str();
    }
protected:
    void SetUp() override {
        ov::element::Type net_type = ov::element::f32;
        auto params = ngraph::builder::makeParams(net_type, { {1, 2}, {1, 2} });
        std::map<std::string, std::string> common_conf, conf;
        size_t layers_number;
        std::tie(targetDevice, common_conf, conf, layers_number) = this->GetParam();

        configuration.insert(common_conf.begin(), common_conf.end());
        configuration.insert(conf.begin(), conf.end());

        auto add_const = ngraph::builder::makeConstant(net_type, ngraph::Shape{1}, std::vector<float>{0.01f});
        auto add_x = std::make_shared<ngraph::opset8::Add>(add_const, params[0]);
        auto add_y = std::make_shared<ngraph::opset8::Add>(add_const, params[1]);

        std::vector<std::shared_ptr<ov::op::v1::Add>> add_nodes_x;
        std::vector<std::shared_ptr<ov::op::v1::Add>> add_nodes_y;
        add_nodes_x.push_back(add_x);
        add_nodes_y.push_back(add_y);

        for (size_t i = 0; i < (layers_number - 2) / 2; ++i) {
            auto add_next_x = std::make_shared<ngraph::opset8::Add>(add_nodes_x.back(), params[0]);
            auto add_next_y = std::make_shared<ngraph::opset8::Add>(add_nodes_y.back(), params[1]);
            add_nodes_x.push_back(add_next_x);
            add_nodes_y.push_back(add_next_y);
        }

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(add_nodes_x.back()),
                                     std::make_shared<ngraph::opset8::Result>(add_nodes_y.back())};

        function = std::make_shared<ngraph::Function>(results, params, "layers_limit");
    }
};

class GNALayersLimit10Test : public GNALayersLimitTest {};
class GNALayersLimit20Test : public GNALayersLimitTest {};
class GNALayersLimit30Test : public GNALayersLimitTest {};

TEST_P(GNALayersLimitTest, CompareWithRefs) {
    Run();
}
TEST_P(GNALayersLimit10Test, CompareWithRefs) {
    Run();
}
TEST_P(GNALayersLimit20Test, CompareWithRefs) {
    Run();
}
TEST_P(GNALayersLimit30Test, CompareWithRefs) {
    Run();
}

std::map<std::string, std::string> common_config {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1"},
    {"GNA_SCALE_FACTOR_1", "1"}
};

std::vector<std::map<std::string, std::string>> configs_20 {
    {{"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
    {{"GNA_COMPILE_TARGET", "GNA_TARGET_2_0"}}
};

std::vector<std::map<std::string, std::string>> configs_30 {
    {{"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_COMPILE_TARGET", "GNA_TARGET_3_0"}}
};

// for GNA v2.0 limit is 4086
std::vector<size_t> layer_limits_20 {4085, 4086, 4087};
// for GNA v3.0 limit is 8191
std::vector<size_t> layer_limits_30 {8190, 8191, 8192};
// small and big values
std::vector<size_t> layer_numbers { 2, 9000 };

INSTANTIATE_TEST_SUITE_P(smoke_GNALimits, GNALayersLimitTest,
                         ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(common_config),
                                ::testing::Values(common_config),
                                ::testing::ValuesIn(layer_numbers)),
                         GNALayersLimitTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GNALimits, GNALayersLimit20Test,
                         ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(common_config),
                                ::testing::ValuesIn(configs_20),
                                ::testing::ValuesIn(layer_limits_20)),
                         GNALayersLimitTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GNALimits, GNALayersLimit30Test,
                         ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(common_config),
                                ::testing::ValuesIn(configs_30),
                                ::testing::ValuesIn(layer_limits_30)),
                         GNALayersLimitTest::getTestCaseName);

} // namespace LayerTestsDefinitions