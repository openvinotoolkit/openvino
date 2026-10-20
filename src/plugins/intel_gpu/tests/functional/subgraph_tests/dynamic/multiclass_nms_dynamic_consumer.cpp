// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using MulticlassNmsConsumerParams = std::tuple<ov::PartialShape,  // boxes
                                               ov::PartialShape>; // scores

class MulticlassNmsDynamicConsumerGPUTest : public testing::TestWithParam<MulticlassNmsConsumerParams> {};

TEST_P(MulticlassNmsDynamicConsumerGPUTest, OutputsFollowRuntimeBoxCount) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [boxes_shape, scores_shape] = GetParam();
    const bool static_inputs = boxes_shape.is_static() && scores_shape.is_static();

    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, boxes_shape);
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scores_shape);
    ov::op::v9::MulticlassNms::Attributes attrs;
    attrs.sort_result_type = ov::op::v9::MulticlassNms::SortResultType::SCORE;
    attrs.output_type = ov::element::i32;
    attrs.iou_threshold = 0.5f;
    auto nms = std::make_shared<ov::op::v9::MulticlassNms>(boxes, scores, attrs);
    auto add = std::make_shared<ov::op::v1::Add>(nms->output(0),
                                                 ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.f}));
    auto convert = std::make_shared<ov::op::v0::Convert>(nms->output(1), ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add, convert, nms->output(2)},
                                             ov::ParameterVector{boxes, scores});

    ov::Core core;
    auto compiled = core.compile_model(model,
                                       ov::test::utils::DEVICE_GPU,
                                       ov::hint::inference_precision(ov::element::f32));
    if (static_inputs) {
        ASSERT_EQ(compiled.output(0).get_partial_shape(), (ov::PartialShape{10, 6}));
        ASSERT_EQ(compiled.output(1).get_partial_shape(), (ov::PartialShape{10, 1}));
    } else {
        ASSERT_TRUE(compiled.output(0).get_partial_shape()[0].is_dynamic());
        ASSERT_TRUE(compiled.output(1).get_partial_shape()[0].is_dynamic());
    }

    auto request = compiled.create_infer_request();
    const std::vector<size_t> box_counts = static_inputs ? std::vector<size_t>{10} : std::vector<size_t>{10, 4, 10, 1};
    for (const size_t num_boxes : box_counts) {
        SCOPED_TRACE(num_boxes);
        ov::Tensor boxes_tensor(ov::element::f32, ov::Shape{1, num_boxes, 4});
        ov::Tensor scores_tensor(ov::element::f32, ov::Shape{1, 1, num_boxes});
        auto boxes_data = boxes_tensor.data<float>();
        auto scores_data = scores_tensor.data<float>();
        for (size_t i = 0; i < num_boxes; ++i) {
            const float x = 2.f * static_cast<float>(i);
            boxes_data[4 * i + 0] = x;
            boxes_data[4 * i + 1] = 0.f;
            boxes_data[4 * i + 2] = x + 1.f;
            boxes_data[4 * i + 3] = 1.f;
            scores_data[i] = static_cast<float>(i + 1) / 16.f;
        }
        request.set_input_tensor(0, boxes_tensor);
        request.set_input_tensor(1, scores_tensor);
        request.infer();

        const auto selected = request.get_output_tensor(2);
        ASSERT_EQ(selected.get_shape(), (ov::Shape{1}));
        ASSERT_EQ(selected.data<int32_t>()[0], static_cast<int32_t>(num_boxes));
        const auto rows = request.get_output_tensor(0);
        const auto indices = request.get_output_tensor(1);
        ASSERT_EQ(rows.get_shape(), (ov::Shape{num_boxes, 6}));
        ASSERT_EQ(indices.get_shape(), (ov::Shape{num_boxes, 1}));
        for (size_t row = 0; row < num_boxes; ++row) {
            // Boxes do not overlap, so all are kept in descending score order.
            const size_t box = num_boxes - 1 - row;
            const float x = 2.f * static_cast<float>(box);
            const std::array<float, 6> expected{0.f, static_cast<float>(box + 1) / 16.f, x, 0.f, x + 1.f, 1.f};
            for (size_t col = 0; col < expected.size(); ++col) {
                EXPECT_FLOAT_EQ(rows.data<float>()[row * 6 + col], expected[col] + 1.f);
            }
            EXPECT_FLOAT_EQ(indices.data<float>()[row], static_cast<float>(box));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsDynamicConsumer_GPU,
                         MulticlassNmsDynamicConsumerGPUTest,
                         testing::Values(MulticlassNmsConsumerParams{{1, {1, 10}, 4}, {1, 1, {1, 10}}},
                                         MulticlassNmsConsumerParams{{-1, -1, 4}, {-1, 1, -1}},
                                         MulticlassNmsConsumerParams{{1, 10, 4}, {1, 1, 10}}));

}  // namespace
