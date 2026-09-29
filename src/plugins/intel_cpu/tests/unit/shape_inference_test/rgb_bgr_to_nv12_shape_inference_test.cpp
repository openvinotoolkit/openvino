// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

template <class TOp>
class ConvertColorToNV12Test : public OpStaticShapeInferenceTest<TOp> {};

TYPED_TEST_SUITE_P(ConvertColorToNV12Test);

TYPED_TEST_P(ConvertColorToNV12Test, single_plane_default) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4, 6, 3});
    this->op = this->make_op(rgb);

    this->input_shapes = StaticShapeVector{{3, 4, 6, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 6, 6, 1}));  // H*3/2 = 4*3/2 = 6
}

TYPED_TEST_P(ConvertColorToNV12Test, single_plane_explicit) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{1, 8, 10, 3});
    this->op = this->make_op(rgb, true);

    this->input_shapes = StaticShapeVector{{1, 8, 10, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 12, 10, 1}));  // H*3/2 = 8*3/2 = 12
}

TYPED_TEST_P(ConvertColorToNV12Test, two_plane) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 480, 640, 3});
    this->op = this->make_op(rgb, false);

    this->input_shapes = StaticShapeVector{{2, 480, 640, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 480, 640, 1}));  // Y plane
    EXPECT_EQ(output_shapes[1], StaticShape({2, 240, 320, 2}));  // UV plane
}

TYPED_TEST_P(ConvertColorToNV12Test, single_plane_dynamic_rank) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    this->op = this->make_op(rgb);

    this->input_shapes = StaticShapeVector{{1, 6, 8, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 9, 8, 1}));  // H*3/2 = 6*3/2 = 9
}

TYPED_TEST_P(ConvertColorToNV12Test, single_plane_static_rank) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(rgb);

    this->input_shapes = StaticShapeVector{{5, 2, 2, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 3, 2, 1}));
}

TYPED_TEST_P(ConvertColorToNV12Test, two_plane_dynamic_rank) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    this->op = this->make_op(rgb, false);

    this->input_shapes = StaticShapeVector{{3, 10, 10, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_EQ(output_shapes[0], StaticShape({3, 10, 10, 1}));
    EXPECT_EQ(output_shapes[1], StaticShape({3, 5, 5, 2}));
}

TYPED_TEST_P(ConvertColorToNV12Test, two_plane_static_rank) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(rgb, false);

    this->input_shapes = StaticShapeVector{{5, 20, 20, 3}};
    auto output_shapes = shape_inference(this->op.get(), this->input_shapes);

    EXPECT_EQ(output_shapes.size(), 2);
    EXPECT_EQ(output_shapes[0], StaticShape({5, 20, 20, 1}));
    EXPECT_EQ(output_shapes[1], StaticShape({5, 10, 10, 2}));
}

TYPED_TEST_P(ConvertColorToNV12Test, two_plane_missing_channel) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(rgb, false);

    this->input_shapes = StaticShapeVector{{5, 20, 20, 2}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("RGB/BGR input number of channels should be equal to 3"));
}

TYPED_TEST_P(ConvertColorToNV12Test, two_plane_height_not_div_by_2) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(rgb, false);

    this->input_shapes = StaticShapeVector{{5, 19, 20, 3}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Image height must be even"));
}

TYPED_TEST_P(ConvertColorToNV12Test, two_plane_width_not_div_by_2) {
    const auto rgb = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    this->op = this->make_op(rgb, false);

    this->input_shapes = StaticShapeVector{{5, 20, 19, 3}};
    OV_EXPECT_THROW(shape_inference(this->op.get(), this->input_shapes),
                    NodeValidationFailure,
                    HasSubstr("Image width must be even"));
}

REGISTER_TYPED_TEST_SUITE_P(ConvertColorToNV12Test,
                            single_plane_default,
                            single_plane_explicit,
                            two_plane,
                            single_plane_dynamic_rank,
                            single_plane_static_rank,
                            two_plane_dynamic_rank,
                            two_plane_static_rank,
                            two_plane_missing_channel,
                            two_plane_height_not_div_by_2,
                            two_plane_width_not_div_by_2);

using ToNV12Types = testing::Types<op::v17::RGBtoNV12, op::v17::BGRtoNV12>;
INSTANTIATE_TYPED_TEST_SUITE_P(StaticShapeInference, ConvertColorToNV12Test, ToNV12Types);
