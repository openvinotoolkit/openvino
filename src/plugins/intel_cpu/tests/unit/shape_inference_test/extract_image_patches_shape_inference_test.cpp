
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "extract_image_patches_shape_inference.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

class StaticShapeExtractImagePatchesV3Test : public OpStaticShapeInferenceTest<op::v3::ExtractImagePatches> {};

TEST_F(StaticShapeExtractImagePatchesV3Test, default_ctor_no_args) {
    auto pad_type = op::PadType::VALID;

    op = make_op();
    op->set_sizes({3, 3});
    op->set_strides({5, 5});
    op->set_rates({1, 1});
    op->set_auto_pad(pad_type);

    input_shapes = StaticShapeVector{{10, 8, 12, 6}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 72, 2, 1}));
}

TEST_F(StaticShapeExtractImagePatchesV3Test, data_input_is_dynamic_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    op = make_op(data, ov::Shape{3, 3}, ov::Strides{5, 5}, ov::Shape{2, 2}, op::PadType::VALID);

    input_shapes = StaticShapeVector{{2, 2, 23, 24}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 18, 4, 4}));
}

TEST_F(StaticShapeExtractImagePatchesV3Test, data_input_is_static_rank) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic(4));
    op = make_op(data, ov::Shape{3, 3}, ov::Strides{5, 5}, ov::Shape{1, 1}, op::PadType::SAME_UPPER);

    input_shapes = StaticShapeVector{{2, 2, 43, 34}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({2, 18, 9, 7}));
}

TEST_F(StaticShapeExtractImagePatchesV3Test, data_shape_not_compatible_rank_4) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic(4));
    op = make_op(data, ov::Shape{3, 3}, ov::Strides{5, 5}, ov::Shape{1, 1}, op::PadType::SAME_UPPER);

    OV_EXPECT_THROW(shape_inference(op.get(), StaticShapeVector{{2, 20, 12, 24, 1}}),
                    NodeValidationFailure,
                    HasSubstr("input tensor must be 4D tensor"));
}
