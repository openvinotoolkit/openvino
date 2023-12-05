// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "openvino/opsets/opset8.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace InferenceEngine;
using namespace ov::opset8;

namespace LayerTestsDefinitions {

using GNALayersLimitTestParams = std::tuple<std::string,                         // Device name
                                            std::map<std::string, std::string>,  // common config
                                            std::map<std::string, std::string>,  // Configuration
                                            size_t>;                             // number of layers

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
        for (auto const& conf_i : conf) {
            result << "_configItem=" << conf_i.first.c_str() << "_" << conf_i.second.c_str();
        }
        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type net_type = ov::element::f32;
        const size_t branch_count = 64;
        std::vector<std::vector<size_t>> shapes(branch_count, {1, 2});

        std::map<std::string, std::string> common_conf, conf;
        size_t layers_number;
        std::tie(targetDevice, common_conf, conf, layers_number) = this->GetParam();

        configuration.insert(common_conf.begin(), common_conf.end());
        configuration.insert(conf.begin(), conf.end());

        ov::ParameterVector params;
        for (auto&& shape : shapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(net_type, ov::Shape(shape)));
        }
        auto add_const = ngraph::builder::makeConstant(net_type, ov::Shape{1}, std::vector<float>{0.01f});
        ov::ResultVector results;

        std::vector<std::vector<std::shared_ptr<Add>>> branches;

        for (size_t i = 0; i < branch_count; ++i) {
            configuration.insert({"GNA_SCALE_FACTOR_" + std::to_string(i), "1"});
            std::vector<std::shared_ptr<Add>> add_nodes;
            add_nodes.push_back(std::make_shared<Add>(add_const, params[i]));

            for (size_t j = 0; j < (layers_number - branch_count) / branch_count; ++j) {
                add_nodes.push_back(std::make_shared<Add>(add_nodes.back(), params[i]));
            }
            branches.push_back(add_nodes);
            results.push_back(std::make_shared<Result>(add_nodes.back()));
        }
        function = std::make_shared<ov::Model>(results, params, "layers_limit");
    };
};

class GNALayersLimit20Test : public GNALayersLimitTest {};
class GNALayersLimit3XTest : public GNALayersLimitTest {};

TEST_P(GNALayersLimit20Test, CompareWithRefs) {
    Run();
}

TEST_P(GNALayersLimit3XTest, CompareWithRefs) {
    Run();
}

std::map<std::string, std::string> common_config{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}};

std::vector<std::map<std::string, std::string>> configs_20{{{"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
                                                           {{"GNA_COMPILE_TARGET", "GNA_TARGET_2_0"}}};

std::vector<std::map<std::string, std::string>> configs_3X{{{"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
                                                           {{"GNA_COMPILE_TARGET", "GNA_TARGET_3_0"}},
                                                           {{"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
                                                           {{"GNA_COMPILE_TARGET", "GNA_TARGET_3_5"}}};

// for GNA v2.0 limit is 4096
std::vector<size_t> layer_limits_20{64, 4096, 4160};
// for GNA >= v3.0 limit is 8192
std::vector<size_t> layer_limits_3X{64, 8192, 8200};

INSTANTIATE_TEST_SUITE_P(smoke_GNALimits,
                         GNALayersLimit20Test,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(common_config),
                                            ::testing::ValuesIn(configs_20),
                                            ::testing::ValuesIn(layer_limits_20)),
                         GNALayersLimitTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GNALimits,
                         GNALayersLimit3XTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(common_config),
                                            ::testing::ValuesIn(configs_3X),
                                            ::testing::ValuesIn(layer_limits_3X)),
                         GNALayersLimitTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
