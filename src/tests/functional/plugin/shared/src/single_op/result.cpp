// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/result.hpp"

namespace ov {
namespace test {
std::string ResultLayerTest::getTestCaseName(const testing::TestParamInfo<ResultTestParamSet>& obj) {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(input_shape, model_type, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ResultLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    std::tie(input_shape, model_type, targetDevice) = GetParam();

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape))};
    auto result = std::make_shared<ov::op::v0::Result>(params[0]);
    function = std::make_shared<ov::Model>(result->outputs(), params, "result");
}
}  // namespace test
}  // namespace ov
