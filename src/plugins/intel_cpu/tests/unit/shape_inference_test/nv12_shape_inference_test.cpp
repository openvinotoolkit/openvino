// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class ConvertColorNV12Test : public OpStaticShapeInferenceTest<TOp> {};

TYPED_TEST_SUITE_P(ConvertColorNV12Test);

TYPED_TEST_P(ConvertColorNV12Test, default_ctor_single_plane_no_args) {
    this->op = this->make_op();

    this->input_shapes = StaticShapeVector{{3, 30, 10, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 20, 10, 3}));
}

TYPED_TEST_P(ConvertColorNV12Test, default_ctor_two_plane_no_args) {
    this->op = this->make_op();

    this->input_shapes = StaticShapeVector{{3, 20, 20, 1}, {3, 10, 10, 2}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 20, 20, 3}));
}

TYPED_TEST_P(ConvertColorNV12Test, single_plane_dynamic_rank) {
    const auto yuv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    this->op = this->make_op(yuv);

    this->input_shapes = StaticShapeVector{{3, 12, 10, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 8, 10, 3}));
}

TYPED_TEST_P(ConvertColorNV12Test, single_plane_static_rank) {
    const auto yuv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(yuv);

    this->input_shapes = StaticShapeVector{{5, 3, 2, 1}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 2, 2, 3}));
}

TYPED_TEST_P(ConvertColorNV12Test, two_plane_dynamic_rank) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto uv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    this->op = this->make_op(y, uv);

    this->input_shapes = StaticShapeVector{{3, 10, 10, 1}, {3, 5, 5, 2}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 10, 10, 3}));
}

TYPED_TEST_P(ConvertColorNV12Test, two_plane_static_rank) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto uv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(y, uv);

    this->input_shapes = StaticShapeVector{{5, 20, 20, 1}, {5, 10, 10, 2}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 20, 20, 3}));
}

TYPED_TEST_P(ConvertColorNV12Test, two_plane_uv_shape_not_compatible) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto uv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(y, uv);

    this->input_shapes = StaticShapeVector{{5, 20, 20, 1}, {4, 10, 10, 2}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Y shape is inconsistent with UV"));
}

TYPED_TEST_P(ConvertColorNV12Test, two_plane_y_dims_not_div_by_2) {
    const auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto uv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(y, uv);

    this->input_shapes = StaticShapeVector{{5, 19, 19, 1}, {4, 10, 10, 2}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Y shape is inconsistent with UV"));
}

TYPED_TEST_P(ConvertColorNV12Test, single_plane_height_not_div_by_three) {
    const auto yuv = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(yuv);

    this->input_shapes = StaticShapeVector{{5, 19, 20, 1}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Image height shall be divisible by 3"));
}

REGISTER_TYPED_TEST_SUITE_P(ConvertColorNV12Test,
                            default_ctor_single_plane_no_args,
                            default_ctor_two_plane_no_args,
                            single_plane_dynamic_rank,
                            single_plane_static_rank,
                            two_plane_dynamic_rank,
                            two_plane_static_rank,
                            two_plane_uv_shape_not_compatible,
                            two_plane_y_dims_not_div_by_2,
                            single_plane_height_not_div_by_three);

using NV12Types = testing::Types<op::v8::NV12toRGB, op::v8::NV12toBGR>;
INSTANTIATE_TYPED_TEST_SUITE_P(StaticShapeInference, ConvertColorNV12Test, NV12Types);
