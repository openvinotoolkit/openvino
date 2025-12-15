// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_compressed_weights.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace test {

std::string MatmulCompressedTest::getTestCaseName(const testing::TestParamInfo<matmulCompressedParams>& obj) {
    const auto& [input_type, weights_type, targetDevice, configuration, input_shape, weights_shape] = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "WS=" << ov::test::utils::vec2str(weights_shape) << "_";
    result << "IT=" << input_type << "_";
    result << "WT=" << weights_type << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second.as<std::string>();
    }
    return result.str();
}

void MatmulCompressedTest::SetUp() {
    const auto& [input_type, weights_type, _targetDevice, tempConfig, inputShape, weights_shape] = this->GetParam();
    targetDevice = _targetDevice;
    configuration.insert(tempConfig.begin(), tempConfig.end());

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape(inputShape))};

    std::vector<uint8_t> weights_data(ov::shape_size(weights_shape), 0x01);
    auto weights = ov::op::v0::Constant::create(weights_type, weights_shape, weights_data);
    auto convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto scale_shape = weights_shape;
    scale_shape.back() = 1;
    auto scale = ov::op::v0::Constant::create(ov::element::f32, scale_shape, std::vector<float>(ov::shape_size(scale_shape), 0.1f));
    auto multiply = std::make_shared<ov::op::v1::Multiply>(convert, scale);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], multiply, false, true);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul)};
    function = std::make_shared<ov::Model>(results, params, "MatmulCompressedTest");
}

}  // namespace test
}  // namespace ov
