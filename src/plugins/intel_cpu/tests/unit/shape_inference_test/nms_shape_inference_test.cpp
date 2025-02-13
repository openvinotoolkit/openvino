// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class NMSNonDynamicOutputTest : public OpStaticShapeInferenceTest<TOp> {};
TYPED_TEST_SUITE_P(NMSNonDynamicOutputTest);

TYPED_TEST_P(NMSNonDynamicOutputTest, default_ctor_no_args) {
    const auto op = this->make_op();

    int16_t max_output_boxes = 3;
    const auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i16, ov::Shape{}, &max_output_boxes}}};

    this->input_shapes = StaticShapeVector{{1, 6, 4}, {1, 1, 6}, {}, {}, {}};
    const auto output_shapes = shape_inference(op.get(), this->input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 3}));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, boxes_scores_dynamic_rank_max_out_as_const) {
    const auto boxes = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = op::v0::Constant::create(element::i16, ov::Shape{}, {3});
    const auto iou_threshold = std::make_shared<op::v0::Parameter>(element::f32, ov::Shape{});
    const auto score_threshold = std::make_shared<op::v0::Parameter>(element::f32, ov::Shape{});

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    this->input_shapes = StaticShapeVector{{1, 6, 4}, {1, 1, 6}, {}, {}, {}};
    const auto output_shapes = shape_inference(op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 3}));
}

TYPED_TEST_P(NMSNonDynamicOutputTest, all_inputs_are_dynamic) {
    const auto boxes = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto max_output_boxes_per_class = std::make_shared<op::v0::Parameter>(element::i16, PartialShape::dynamic());
    const auto iou_threshold = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto score_threshold = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());

    int16_t max_output_boxes = 3;
    const auto const_data = std::unordered_map<size_t, Tensor>{{2, {element::i16, ov::Shape{}, &max_output_boxes}}};

    const auto op = this->make_op(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

    this->input_shapes = StaticShapeVector{{1, 6, 4}, {1, 1, 6}, {}, {}, {}};
    const auto output_shapes = shape_inference(op.get(), this->input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 3}));
}

REGISTER_TYPED_TEST_SUITE_P(NMSNonDynamicOutputTest,
                            default_ctor_no_args,
                            boxes_scores_dynamic_rank_max_out_as_const,
                            all_inputs_are_dynamic);

using NMSNonDynamicOutputTypes =
    testing::Types<op::v1::NonMaxSuppression, op::v3::NonMaxSuppression, op::v4::NonMaxSuppression>;
INSTANTIATE_TYPED_TEST_SUITE_P(StaticShapeInference, NMSNonDynamicOutputTest, NMSNonDynamicOutputTypes);
