// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class ConvertColorI420Test : public OpStaticShapeInferenceTest<TOp> {
protected:
};

TYPED_TEST_SUITE_P(ConvertColorI420Test);

TYPED_TEST_P(ConvertColorI420Test, default_ctor_single_plane_no_args) {
    this->op = this->make_op();

    this->input_shapes = StaticShapeVector{{3, 15, 10, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 10, 10, 3}));
}

TYPED_TEST_P(ConvertColorI420Test, default_ctor_three_plane_no_args) {
    this->op = this->make_op();

    this->input_shapes = StaticShapeVector{{3, 20, 20, 1}, {3, 10, 10, 1}, {3, 10, 10, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 20, 20, 3}));
}

TYPED_TEST_P(ConvertColorI420Test, single_plane_dynamic_rank) {
    const auto yuv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    this->op = this->make_op(yuv);

    this->input_shapes = StaticShapeVector{{3, 12, 10, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 8, 10, 3}));
}

TYPED_TEST_P(ConvertColorI420Test, single_plane_static_rank) {
    const auto yuv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(yuv);

    this->input_shapes = StaticShapeVector{{5, 3, 2, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 2, 2, 3}));
}

TYPED_TEST_P(ConvertColorI420Test, three_plane_dynamic_rank) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto u = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto v = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    this->op = this->make_op(y, u, v);

    this->input_shapes = StaticShapeVector{{3, 10, 10, 1}, {3, 5, 5, 1}, {3, 5, 5, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 10, 10, 3}));
}

TYPED_TEST_P(ConvertColorI420Test, three_plane_static_rank) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto u = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto v = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(y, u, v);

    this->input_shapes = StaticShapeVector{{5, 20, 20, 1}, {5, 10, 10, 1}, {5, 10, 10, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 20, 20, 3}));
}

TYPED_TEST_P(ConvertColorI420Test, three_plane_u_shape_not_compatible) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto u = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto v = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(y, u, v);

    this->input_shapes = StaticShapeVector{{5, 20, 20, 1}, {4, 10, 10, 1}, {5, 10, 10, 1}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Y shape is inconsistent with U and V"));
}

TYPED_TEST_P(ConvertColorI420Test, single_plane_height_not_div_by_three) {
    const auto yuv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(yuv);

    this->input_shapes = StaticShapeVector{{5, 19, 20, 1}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Image height shall be divisible by 3"));
}

REGISTER_TYPED_TEST_SUITE_P(ConvertColorI420Test,
                            default_ctor_single_plane_no_args,
                            default_ctor_three_plane_no_args,
                            single_plane_dynamic_rank,
                            single_plane_static_rank,
                            three_plane_dynamic_rank,
                            three_plane_static_rank,
                            three_plane_u_shape_not_compatible,
                            single_plane_height_not_div_by_three);

using I420Types = testing::Types<op::v8::I420toRGB, op::v8::I420toBGR>;
INSTANTIATE_TYPED_TEST_SUITE_P(StaticShapeInference, ConvertColorI420Test, I420Types);
