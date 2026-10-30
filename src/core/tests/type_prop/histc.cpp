// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov::test {
using op::v0::Parameter;
using testing::HasSubstr;

class TypePropHistcTest : public TypePropOpTest<op::v17::Histc> {};

TEST_F(TypePropHistcTest, default_ctor) {
    const auto op = make_op();
    op->set_argument(0, std::make_shared<Parameter>(element::f32, PartialShape{2, 3}));
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_bins(), 100);
    EXPECT_EQ(op->get_min(), 0.0);
    EXPECT_EQ(op->get_max(), 0.0);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{100}));
}

TEST_F(TypePropHistcTest, output_shape_does_not_depend_on_input_shape) {
    for (const auto& shape :
         {PartialShape{2, 3, 4}, PartialShape{}, PartialShape{0}, PartialShape{-1, {2, 5}}, PartialShape::dynamic()}) {
        const auto op = make_op(std::make_shared<Parameter>(element::f16, shape), 4, 0.0, 3.0);
        EXPECT_EQ(op->get_output_element_type(0), element::f16);
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4})) << shape;
    }
}

TEST_F(TypePropHistcTest, element_types) {
    for (const auto& et : {element::f32, element::f64, element::bf16, element::i32, element::u8, element::dynamic}) {
        const auto op = make_op(std::make_shared<Parameter>(et, PartialShape{5}), 3);
        EXPECT_EQ(op->get_output_element_type(0), et);
    }
}

TEST_F(TypePropHistcTest, invalid_element_type) {
    const auto data = std::make_shared<Parameter>(element::boolean, PartialShape{5});
    OV_EXPECT_THROW(std::ignore = make_op(data), NodeValidationFailure, HasSubstr("must be numeric"));
}

TEST_F(TypePropHistcTest, invalid_bins) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{5});
    OV_EXPECT_THROW(std::ignore = make_op(data, 0), NodeValidationFailure, HasSubstr("'bins' must be positive"));
    OV_EXPECT_THROW(std::ignore = make_op(data, -3), NodeValidationFailure, HasSubstr("'bins' must be positive"));
}

TEST_F(TypePropHistcTest, invalid_range) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{5});
    const auto inf = std::numeric_limits<double>::infinity();
    OV_EXPECT_THROW(std::ignore = make_op(data, 4, 2.0, 1.0),
                    NodeValidationFailure,
                    HasSubstr("'min' must be less or equal than 'max'"));
    OV_EXPECT_THROW(std::ignore = make_op(data, 4, -inf, 1.0), NodeValidationFailure, HasSubstr("must be finite"));
    OV_EXPECT_THROW(std::ignore = make_op(data, 4, 0.0, std::numeric_limits<double>::quiet_NaN()),
                    NodeValidationFailure,
                    HasSubstr("must be finite"));
    OV_EXPECT_THROW(std::ignore = make_op(data, 4, 0.0, inf), NodeValidationFailure, HasSubstr("must be finite"));
}

TEST_F(TypePropHistcTest, valid_attribute_edges) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{5});
    EXPECT_EQ(make_op(data, 1, 5.0, 5.0)->get_output_partial_shape(0), (PartialShape{1}));
    EXPECT_EQ(make_op(data, 3, -3.0, -1.0)->get_output_partial_shape(0), (PartialShape{3}));
    EXPECT_EQ(make_op(data, int64_t{1} << 20, -1e300, 1e300)->get_output_partial_shape(0),
              (PartialShape{int64_t{1} << 20}));
}

TEST_F(TypePropHistcTest, setters_are_validated) {
    const auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{5}), 4, 0.0, 1.0);
    op->set_bins(6);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6}));

    op->set_min(2.0);
    OV_EXPECT_THROW(op->validate_and_infer_types(),
                    NodeValidationFailure,
                    HasSubstr("'min' must be less or equal than 'max'"));
    op->set_min(0.0);
    op->set_bins(0);
    OV_EXPECT_THROW(op->validate_and_infer_types(), NodeValidationFailure, HasSubstr("'bins' must be positive"));
}

TEST_F(TypePropHistcTest, clone_preserves_attributes) {
    const auto op = make_op(std::make_shared<Parameter>(element::f32, PartialShape{5}), 7, -1.5, 2.5);
    const auto new_data = std::make_shared<Parameter>(element::i32, PartialShape{-1, 3});
    const auto clone = ov::as_type_ptr<op::v17::Histc>(op->clone_with_new_inputs({new_data}));

    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_bins(), 7);
    EXPECT_EQ(clone->get_min(), -1.5);
    EXPECT_EQ(clone->get_max(), 2.5);
    EXPECT_EQ(clone->get_output_element_type(0), element::i32);
    EXPECT_EQ(clone->get_output_partial_shape(0), (PartialShape{7}));
}
}  // namespace ov::test
