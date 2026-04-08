// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouped_matmul.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov::test {
using op::v0::Constant, op::v0::Parameter;
using testing::HasSubstr;

class TypePropGroupedMatMulTest : public TypePropOpTest<op::v17::GroupedMatMul> {};

// ==================== Case 1: 2D × 3D (MoE forward) ====================

TEST_F(TypePropGroupedMatMulTest, case1_2d_3d_basic) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, 128});
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{3});

    const auto op = make_op(mat_a, mat_b, offsets);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{16, 128}));
}

TEST_F(TypePropGroupedMatMulTest, case1_2d_3d_dynamic_rows) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, 128});
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{3});

    const auto op = make_op(mat_a, mat_b, offsets);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 128}));
}

TEST_F(TypePropGroupedMatMulTest, case1_2d_3d_f16) {
    const auto mat_a = std::make_shared<Parameter>(element::f16, PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f16, PartialShape{3, 64, 128});
    const auto offsets = std::make_shared<Parameter>(element::i64, PartialShape{3});

    const auto op = make_op(mat_a, mat_b, offsets);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{16, 128}));
}

TEST_F(TypePropGroupedMatMulTest, case1_2d_3d_inner_dim_mismatch) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 32, 128});  // K=32 != 64
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{3});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b, offsets),
                    ov::NodeValidationFailure,
                    HasSubstr("inner dimension mismatch"));
}

TEST_F(TypePropGroupedMatMulTest, case1_2d_3d_missing_offsets) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, 128});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b),
                    ov::NodeValidationFailure,
                    HasSubstr("requires offsets"));
}

// ==================== Case 2: 3D × 3D (batched) ====================

TEST_F(TypePropGroupedMatMulTest, case2_3d_3d_basic) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{3, 4, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, 128});

    const auto op = make_op(mat_a, mat_b);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4, 128}));
}

TEST_F(TypePropGroupedMatMulTest, case2_3d_3d_bf16) {
    const auto mat_a = std::make_shared<Parameter>(element::bf16, PartialShape{8, 4, 512});
    const auto mat_b = std::make_shared<Parameter>(element::bf16, PartialShape{8, 512, 2048});

    const auto op = make_op(mat_a, mat_b);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{8, 4, 2048}));
}

TEST_F(TypePropGroupedMatMulTest, case2_3d_3d_dynamic_batch) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), 64, 128});

    const auto op = make_op(mat_a, mat_b);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 4, 128}));
}

TEST_F(TypePropGroupedMatMulTest, case2_3d_3d_batch_mismatch) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{3, 4, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{5, 64, 128});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b),
                    ov::NodeValidationFailure,
                    HasSubstr("batch dimension mismatch"));
}

TEST_F(TypePropGroupedMatMulTest, case2_3d_3d_inner_dim_mismatch) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{3, 4, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 32, 128});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b),
                    ov::NodeValidationFailure,
                    HasSubstr("inner dimension mismatch"));
}

// ==================== Case 3: 2D × 2D (weight gradient) ====================

TEST_F(TypePropGroupedMatMulTest, case3_2d_2d_basic) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{64, 16});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{16, 128});
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{3});

    const auto op = make_op(mat_a, mat_b, offsets);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 64, 128}));
}

TEST_F(TypePropGroupedMatMulTest, case3_2d_2d_dynamic_groups) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{64, 16});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{16, 128});
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    const auto op = make_op(mat_a, mat_b, offsets);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 64, 128}));
}

TEST_F(TypePropGroupedMatMulTest, case3_2d_2d_shared_dim_mismatch) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{64, 16});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{20, 128});  // 20 != 16
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{3});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b, offsets),
                    ov::NodeValidationFailure,
                    HasSubstr("shared dimension mismatch"));
}

TEST_F(TypePropGroupedMatMulTest, case3_2d_2d_missing_offsets) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{64, 16});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{16, 128});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b),
                    ov::NodeValidationFailure,
                    HasSubstr("requires offsets"));
}

// ==================== Error cases ====================

TEST_F(TypePropGroupedMatMulTest, unsupported_ndim_combination) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{3, 4, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{64, 128});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b),
                    ov::NodeValidationFailure,
                    HasSubstr("unsupported combination"));
}

TEST_F(TypePropGroupedMatMulTest, dtype_mismatch) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{3, 4, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f16, PartialShape{3, 64, 128});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b),
                    ov::NodeValidationFailure,
                    HasSubstr("do not have the same element type"));
}

TEST_F(TypePropGroupedMatMulTest, invalid_offsets_dtype) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, 128});
    const auto offsets = std::make_shared<Parameter>(element::f32, PartialShape{3});

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b, offsets),
                    ov::NodeValidationFailure,
                    HasSubstr("Offsets element type must be i32 or i64"));
}

TEST_F(TypePropGroupedMatMulTest, offsets_not_1d) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, 128});
    const auto offsets = std::make_shared<Parameter>(element::i32, PartialShape{3, 2});  // 2D, not 1D

    OV_EXPECT_THROW(std::ignore = make_op(mat_a, mat_b, offsets),
                    ov::NodeValidationFailure,
                    HasSubstr("offsets must be 1D"));
}

// ==================== Dynamic shapes ====================

TEST_F(TypePropGroupedMatMulTest, fully_dynamic_shapes) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    const auto op = make_op(mat_a, mat_b);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropGroupedMatMulTest, case2_3d_3d_partial_dynamic) {
    const auto mat_a = std::make_shared<Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 64});
    const auto mat_b = std::make_shared<Parameter>(element::f32, PartialShape{3, 64, Dimension::dynamic()});

    const auto op = make_op(mat_a, mat_b);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, Dimension::dynamic(), Dimension::dynamic()}));
}

}  // namespace ov::test
