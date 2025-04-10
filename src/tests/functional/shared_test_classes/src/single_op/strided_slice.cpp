// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/strided_slice.hpp"

namespace ov {
namespace test {
std::string StridedSliceLayerTest::getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj) {
    StridedSliceSpecificParams params;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(params, model_type, target_device) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < params.input_shape.size(); i++) {
        result << ov::test::utils::partialShape2str({params.input_shape[i].first})
               << (i < params.input_shape.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < params.input_shape.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < params.input_shape.size(); j++) {
            result << ov::test::utils::vec2str(params.input_shape[j].second[i]) << (j < params.input_shape.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "modelType=" << model_type.to_string() << "_";
    result << "begin=" << ov::test::utils::vec2str(params.begin) << "_";
    result << "end=" << ov::test::utils::vec2str(params.end) << "_";
    result << "stride=" << ov::test::utils::vec2str(params.strides) << "_";
    result << "begin_m=" << ov::test::utils::vec2str(params.begin_mask) << "_";
    result << "end_m=" << ov::test::utils::vec2str(params.end_mask) << "_";
    result << "new_axis_m=" << (params.new_axis_mask.empty() ? "def" : ov::test::utils::vec2str(params.new_axis_mask)) << "_";
    result << "shrink_m=" << (params.shrink_axis_mask.empty() ? "def" : ov::test::utils::vec2str(params.shrink_axis_mask)) << "_";
    result << "ellipsis_m=" << (params.ellipsis_axis_mask.empty() ? "def" : ov::test::utils::vec2str(params.ellipsis_axis_mask)) << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void StridedSliceLayerTest::SetUp() {
    StridedSliceSpecificParams ssParams;
    ov::element::Type model_type;
    std::tie(ssParams, model_type, targetDevice) = this->GetParam();

    init_input_shapes(ssParams.input_shape);

    ASSERT_EQ(ssParams.begin.size(), ssParams.end.size());
    ASSERT_EQ(ssParams.begin.size(), ssParams.strides.size());

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    ov::Shape const_shape = {ssParams.begin.size()};
    auto begin_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, ssParams.begin.data());
    auto end_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, ssParams.end.data());
    auto stride_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, ssParams.strides.data());
    auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(param,
                                                                   begin_node,
                                                                   end_node,
                                                                   stride_node,
                                                                   ssParams.begin_mask,
                                                                   ssParams.end_mask,
                                                                   ssParams.new_axis_mask,
                                                                   ssParams.shrink_axis_mask,
                                                                   ssParams.ellipsis_axis_mask);

    auto result = std::make_shared<ov::op::v0::Result>(stridedSlice);
    function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "StridedSlice");
}
}  // namespace test
}  // namespace ov
