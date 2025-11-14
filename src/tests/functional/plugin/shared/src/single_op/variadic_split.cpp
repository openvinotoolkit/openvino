// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/variadic_split.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace test {
std::string VariadicSplitLayerTest::getTestCaseName(const testing::TestParamInfo<VariadicSplitParams>& obj) {
    const auto& [num_splits, axis, model_type, input_shapes, target_device] = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "numSplits=" << ov::test::utils::vec2str(num_splits) << "_";
    result << "axis=" << axis << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void VariadicSplitLayerTest::SetUp() {
    const auto& [num_splits, axis, model_type, input_shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto split_axis_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{axis});
    auto num_split_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{num_splits.size()}, num_splits);
    auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(param, split_axis_const, num_split_const);
    function = std::make_shared<ov::Model>(variadic_split->outputs(), ov::ParameterVector{param}, "VariadicSplit");
}
}  // namespace test
}  // namespace ov
