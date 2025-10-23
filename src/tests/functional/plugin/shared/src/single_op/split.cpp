// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/split.hpp"
#include "openvino/op/split.hpp"

namespace ov {
namespace test {
std::string SplitLayerTest::getTestCaseName(const testing::TestParamInfo<splitParams>& obj) {
    const auto& [num_splits, axis, model_type, input_shapes, out_indices, target_device] = obj.param;
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
    result << "numSplits=" << num_splits << "_";
    result << "axis=" << axis << "_";
    if (!out_indices.empty()) {
        result << "outIndices" << ov::test::utils::vec2str(out_indices) << "_";
    }
    result << "IS";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void SplitLayerTest::SetUp() {
    const auto& [num_splits, axis, model_type, input_shapes, _out_indices, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;
    auto out_indices = _out_indices;
    if (out_indices.empty()) {
        for (int i = 0; i < num_splits; ++i)
            out_indices.push_back(i);
    }
    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto split_axis_op =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{axis});
    auto splitNode = std::make_shared<ov::op::v1::Split>(param, split_axis_op, num_splits);
    function = std::make_shared<ov::Model>(splitNode->outputs(), ov::ParameterVector{param}, "Split");
}
}  // namespace test
}  // namespace ov
