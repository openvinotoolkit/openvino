// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gguf_dequantize.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov {
namespace test {

using ov::op::internal::GGUFDequantize;
using ov::op::v0::Parameter;
using testing::HasSubstr;

using TypePropGGUFDequantizeTest = TypePropOpTest<GGUFDequantize>;

TEST_F(TypePropGGUFDequantizeTest, q4_k_static_shape) {
    const auto data = std::make_shared<Parameter>(element::u8, PartialShape{64, 2 * 144});
    const auto op = make_op(data, GGUFDequantize::Format::Q4_K, element::f16);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{64, 512}));
}

TEST_F(TypePropGGUFDequantizeTest, q5_k_q6_k_row_bytes) {
    const auto q5 =
        make_op(std::make_shared<Parameter>(element::u8, PartialShape{3, 176}), GGUFDequantize::Format::Q5_K);
    EXPECT_EQ(q5->get_output_partial_shape(0), (PartialShape{3, 256}));
    EXPECT_EQ(q5->get_output_element_type(0), element::f32);
    const auto q6 =
        make_op(std::make_shared<Parameter>(element::u8, PartialShape{2, 5, 4 * 210}), GGUFDequantize::Format::Q6_K);
    EXPECT_EQ(q6->get_output_partial_shape(0), (PartialShape{2, 5, 1024}));
}

TEST_F(TypePropGGUFDequantizeTest, dynamic_dims) {
    const auto rows =
        make_op(std::make_shared<Parameter>(element::u8, PartialShape{-1, 288}), GGUFDequantize::Format::Q4_K);
    EXPECT_EQ(rows->get_output_partial_shape(0), (PartialShape{-1, 512}));
    const auto bytes =
        make_op(std::make_shared<Parameter>(element::u8, PartialShape{8, -1}), GGUFDequantize::Format::Q4_K);
    EXPECT_EQ(bytes->get_output_partial_shape(0), (PartialShape{8, -1}));
    const auto rank =
        make_op(std::make_shared<Parameter>(element::u8, PartialShape::dynamic()), GGUFDequantize::Format::Q6_K);
    EXPECT_EQ(rank->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropGGUFDequantizeTest, invalid_inputs) {
    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::i8, PartialShape{4, 144}),
                                          GGUFDequantize::Format::Q4_K),
                    NodeValidationFailure,
                    HasSubstr("must be u8"));
    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::u8, PartialShape{4, 150}),
                                          GGUFDequantize::Format::Q4_K),
                    NodeValidationFailure,
                    HasSubstr("whole number of"));
    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::u8, PartialShape{144}),
                                          GGUFDequantize::Format::Q4_K),
                    NodeValidationFailure,
                    HasSubstr("rank 2 or more"));
    OV_EXPECT_THROW(std::ignore = make_op(std::make_shared<Parameter>(element::u8, PartialShape{4, 144}),
                                          GGUFDequantize::Format::Q4_K,
                                          element::i32),
                    NodeValidationFailure,
                    HasSubstr("f32, f16 or bf16"));
}

}  // namespace test
}  // namespace ov
