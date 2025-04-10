// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/space_to_batch.hpp"

namespace ov {
namespace test {
std::string SpaceToBatchLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToBatchParamsTuple> &obj) {
    std::vector<InputShape> input_shapes;
    std::vector<int64_t> block_shapes, pads_begin, pads_end;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(block_shapes, pads_begin, pads_end, input_shapes, model_type, target_device) = obj.param;
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
    result << "modelType=" << model_type.to_string() << "_";
    result << "BS=" << ov::test::utils::vec2str(block_shapes) << "_";
    result << "PB=" << ov::test::utils::vec2str(pads_begin) << "_";
    result << "PE=" << ov::test::utils::vec2str(pads_end) << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void SpaceToBatchLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    std::vector<int64_t> block_shapes, pads_begin, pads_end;
    ov::element::Type model_type;
    std::tie(block_shapes, pads_begin, pads_end, input_shapes, model_type, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    ov::Shape const_shape = {param->get_partial_shape().size()};

    ASSERT_EQ(shape_size(const_shape), block_shapes.size());
    ASSERT_EQ(shape_size(const_shape), pads_begin.size());
    ASSERT_EQ(shape_size(const_shape), pads_end.size());

    auto block_shapes_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, block_shapes.data());
    auto pads_begin_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, pads_begin.data());
    auto pads_end_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, pads_end.data());
    auto s2b = std::make_shared<ov::op::v1::SpaceToBatch>(param, block_shapes_node, pads_begin_node, pads_end_node);
    function = std::make_shared<ov::Model>(s2b->outputs(), ov::ParameterVector{param}, "SpaceToBatch");
}
}  // namespace test
}  // namespace ov
