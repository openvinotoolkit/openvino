// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_transpose_to_reshape.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace test {

std::string MatMulTransposeToReshape::getTestCaseName(const testing::TestParamInfo<MatMulTransposeToReshapeParams>& obj) {
    const auto& [element_type, target_name, config] = obj.param;

    std::ostringstream result;
    result << "ET=" << element_type << "_";
    result << "targetDevice=" << target_name;
    for (const auto& item : config) {
        result << "_configItem=" << item.first << "_" << item.second.as<std::string>();
    }
    return result.str();
}

void MatMulTransposeToReshape::SetUp() {
    const auto& [element_type, _targetDevice, additional_config] = GetParam();

    targetDevice = _targetDevice;
    configuration.insert(additional_config.begin(), additional_config.end());

    const auto data = std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape{3, 1, 2});
    const auto weights = ov::test::utils::make_constant(element_type, ov::Shape{1, 2, 1});
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(data, weights, false, false);

    function = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{data}, "MatMulTransposeToReshape");
}

}  // namespace test
}  // namespace ov
