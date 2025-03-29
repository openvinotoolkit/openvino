// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_squeeze_add.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace test {

std::string MatmulSqueezeAddTest::getTestCaseName(const testing::TestParamInfo<matmulSqueezeAddParams>& obj) {
    ov::element::Type element_type;
    ov::Shape input_shape;
    std::size_t outputSize;
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(element_type, targetDevice, configuration, input_shape, outputSize) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "OS=" << outputSize << "_";
    result << "IT=" << element_type << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second.as<std::string>();
    }
    return result.str();
}

void MatmulSqueezeAddTest::SetUp() {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    ov::element::Type element_type;
    ov::AnyMap tempConfig;
    ov::Shape inputShape;
    size_t outputSize;
    std::tie(element_type, targetDevice, tempConfig, inputShape, outputSize) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShape))};

    auto constant_0 = ov::op::v0::Constant::create(element_type, ov::Shape{outputSize, inputShape[1]},
        ov::test::utils::generate_float_numbers(outputSize * inputShape[1], 0, 1, seed));
    auto matmul_0 = std::make_shared<ov::op::v0::MatMul>(params[0], constant_0, false, true);

    auto constant_1 =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{1}, std::vector<size_t>{0});
    auto unsqueeze_0 = std::make_shared<ov::op::v0::Unsqueeze>(matmul_0, constant_1);

    auto constant_2 = ov::op::v0::Constant::create(element_type, ov::Shape{1, inputShape[0], outputSize},
        ov::test::utils::generate_float_numbers(inputShape[0] * outputSize, 0, 1, seed));
    auto add_0 = std::make_shared<ov::op::v1::Add>(unsqueeze_0, constant_2);

    auto constant_3 =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{1}, std::vector<size_t>{0});
    auto squeeze_0 = std::make_shared<ov::op::v0::Squeeze>(add_0, constant_3);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(squeeze_0)};
    function = std::make_shared<ov::Model>(results, params, "MatmulSqueezeAddTest");
}

}  // namespace test
}  // namespace ov
