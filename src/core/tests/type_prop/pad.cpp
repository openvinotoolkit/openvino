// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pad.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;
using namespace ov;
using namespace testing;

template <class T>
class PadTest : public TypePropOpTest<T> {};

TYPED_TEST_SUITE_P(PadTest);

TYPED_TEST_P(PadTest, pad_default_ctor) {
    const auto arg_shape = PartialShape{{1, 2}, {4, 10}, {3, 8}, {1, 2}};
    const auto arg = make_shared<op::v0::Parameter>(element::f32, arg_shape);
    const auto pads_begin = make_shared<op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 2, 1, 0});
    const auto pads_end = make_shared<op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 1, 0});

    const auto pad = this->make_op();
    pad->set_arguments(OutputVector{arg, pads_begin, pads_end});
    pad->set_pad_mode(op::PadMode::REFLECT);
    pad->validate_and_infer_types();

    EXPECT_EQ(pad->get_output_element_type(0), element::f32);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({{1, 2}, {7, 13}, {5, 10}, {1, 2}}));
}

TYPED_TEST_P(PadTest, pad_arg_pad_value_type_mismatch) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto arg_pad_value = make_shared<op::v0::Parameter>(element::f16, Shape{1});

    OV_EXPECT_THROW(ignore = this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT),
                    NodeValidationFailure,
                    HasSubstr("Argument element types do not match (input arg element type:"));
}

TYPED_TEST_P(PadTest, pad_arg_pad_value_shape_not_compatible) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto arg_pad_value = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    OV_EXPECT_THROW(ignore = this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT),
                    NodeValidationFailure,
                    HasSubstr("Argument for padding value is not a scalar"));
}

TYPED_TEST_P(PadTest, pad_pads_begin_shape_not_1D) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{1, 2});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{1});

    OV_EXPECT_THROW(ignore = this->make_op(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC),
                    NodeValidationFailure,
                    HasSubstr("Argument for pads_begin is not 1D"));
}

TYPED_TEST_P(PadTest, pad_pads_end_shape_not_1D) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{1, 2});

    OV_EXPECT_THROW(ignore = this->make_op(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC),
                    NodeValidationFailure,
                    HasSubstr("Argument for pads_end is not 1D"));
}

TYPED_TEST_P(PadTest, pad_pads_begin_size_not_correct) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{1});

    OV_EXPECT_THROW(ignore = this->make_op(arg, pads_begin, pads_end, op::PadMode::SYMMETRIC),
                    NodeValidationFailure,
                    HasSubstr("Number of elements of pads_begin must be >= 0 and <= arg rank"));
}

TYPED_TEST_P(PadTest, pad_pads_end_size_not_correct) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto arg_pad_value = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(ignore = this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT),
                    NodeValidationFailure,
                    HasSubstr("Number of elements of pads_end must be >= 0 and <= arg rank"));
}

TYPED_TEST_P(PadTest, pad_arg_pads_begin_incompatible_type) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i64, Shape{1});

    OV_EXPECT_THROW(this->make_op(arg, pads_begin, pads_end, op::PadMode::REFLECT),
                    NodeValidationFailure,
                    HasSubstr("pads_begin must be an integral number, but is:"));
}

TYPED_TEST_P(PadTest, pad_arg_pads_end_incompatible_type) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::f32, Shape{1});

    OV_EXPECT_THROW(this->make_op(arg, pads_begin, pads_end, op::PadMode::REFLECT),
                    NodeValidationFailure,
                    HasSubstr("pads_end must be an integral number, but is:"));
}

TYPED_TEST_P(PadTest, pad_deduce_too_small_for_edge) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 5, 0, 2});
    auto pads_begin = make_shared<op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto pads_end = make_shared<op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto arg_pad_value = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::EDGE),
                    NodeValidationFailure,
                    HasSubstr("EDGE padding mode requires an input of dimension of at least 1 at each spatial axis"));
}

TYPED_TEST_P(PadTest, pad_deduce_too_small_for_reflect) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{1, 5, 1, 2});
    auto pads_begin = make_shared<op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto pads_end = make_shared<op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto arg_pad_value = make_shared<op::v0::Parameter>(element::f32, Shape{});

    OV_EXPECT_THROW(
        this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::REFLECT),
        NodeValidationFailure,
        HasSubstr("REFLECT padding mode requires an input of dimension of at least 2 at each spatial axis"));
}

TYPED_TEST_P(PadTest, pad_pads_end_got_negative_value) {
    auto arg_shape = PartialShape{-1, {0, 10}, {2, -1}, {2, 8}, {3, 10}, 5};
    auto symbols = set_shape_symbols(arg_shape);
    const auto arg = std::make_shared<op::v0::Parameter>(element::f32, arg_shape);
    const auto pads_begin = op::v0::Constant::create(element::i64, Shape{6}, {2, 0, 1, 3, 2, 1});
    const auto pads_end = op::v0::Constant::create(element::i64, Shape{6}, {-3, -2, -2, -3, -1, -3});

    const auto pad = this->make_op(arg, pads_begin, pads_end, op::PadMode::REFLECT);

    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({-1, {0, 8}, {1, -1}, {2, 8}, {4, 11}, 3}));
    EXPECT_THAT(get_shape_symbols(pad->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, nullptr, symbols[3], nullptr, nullptr));
}

TYPED_TEST_P(PadTest, pad_pads_begin_got_negative_value) {
    auto arg_shape = PartialShape{-1, {0, 10}, {2, -1}, {2, 8}, {3, 10}, 5};
    auto symbols = set_shape_symbols(arg_shape);
    const auto arg = std::make_shared<op::v0::Parameter>(element::f32, arg_shape);
    const auto pads_begin = op::v0::Constant::create(element::i64, Shape{6}, {-1, -1, -2, -3, -8, -4});
    const auto pads_end = op::v0::Constant::create(element::i64, Shape{6}, {0, 2, 0, 3, 5, 4});

    const auto pad = make_shared<TypeParam>(arg, pads_begin, pads_end, op::PadMode::REFLECT);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({-1, {1, 11}, {0, -1}, {2, 8}, {0, 7}, 5}));
    EXPECT_THAT(get_shape_symbols(pad->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, nullptr, symbols[3], nullptr, symbols[5]));
}

TYPED_TEST_P(PadTest, pad_dynamic_output_with_dynamic_rank) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto pads_begin = make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto arg_pad_value = op::v0::Constant::create(element::f32, Shape{}, {0});

    auto pad = this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);
    ASSERT_EQ(pad->get_output_partial_shape(0), PartialShape::dynamic());
}

TYPED_TEST_P(PadTest, pad_dynamic_output_with_static_rank) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto pads_end = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto arg_pad_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0});

    auto pad = make_shared<TypeParam>(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);
    ASSERT_EQ(pad->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TYPED_TEST_P(PadTest, pad_any_dim_for_padding_reflect) {
    auto arg_shape = PartialShape{1, {23, 48}, {23, 48}, 1};
    auto symbols = set_shape_symbols(arg_shape);
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 1, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 1, 0});

    auto pad = make_shared<TypeParam>(arg, pads_begin, pads_end, op::PadMode::REFLECT);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({1, {25, 50}, {25, 50}, 1}));
    EXPECT_THAT(get_shape_symbols(pad->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, symbols[3]));
}

TYPED_TEST_P(PadTest, pad_any_dim_for_padding_edge) {
    auto arg_shape = PartialShape{1, {0, 48}, -1, {20, -1}, {5, -1}, 10, 12};
    auto symbols = set_shape_symbols(arg_shape);
    auto arg = make_shared<op::v0::Parameter>(element::f32, arg_shape);
    auto pads_begin = make_shared<op::v0::Constant>(element::i64, Shape{7}, std::vector<int64_t>{1, 2, 1, 2, 0, 0, 0});
    auto pads_end = make_shared<op::v0::Constant>(element::i64, Shape{7}, std::vector<int64_t>{0, 3, 0, 1, 0, 5, 0});

    auto pad = this->make_op(arg, pads_begin, pads_end, op::PadMode::EDGE);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({2, {5, 53}, {1, -1}, {23, -1}, {5, -1}, 15, 12}));
    EXPECT_THAT(get_shape_symbols(pad->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, nullptr, nullptr, symbols[4], nullptr, symbols[6]));
}

TYPED_TEST_P(PadTest, pad_dynamic_input_type_with_static_value) {
    auto arg = make_shared<op::v0::Parameter>(element::dynamic, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto pads_end = make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto arg_pad_value = op::v0::Constant::create(element::f32, Shape{}, {0});

    auto pad = this->make_op(arg, pads_begin, pads_end, arg_pad_value, op::PadMode::CONSTANT);
    EXPECT_EQ(pad->get_output_element_type(0), element::f32);
    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TYPED_TEST_P(PadTest, pad_preserve_partial_values_and_symbols_via_evaluates_bounds) {
    auto arg_shape = PartialShape{1, {2, 5}, {1, 3}};
    auto begin_shape = PartialShape{{2, 4}, 0, {0, 2}};
    auto end_shape = PartialShape{{1, 2}, 0, 1};
    auto symbols = set_shape_symbols(arg_shape);
    set_shape_symbols(begin_shape);
    set_shape_symbols(end_shape);

    auto arg = make_shared<op::v0::Parameter>(element::f32, arg_shape);
    auto s_begin = make_shared<op::v0::ShapeOf>(make_shared<op::v0::Parameter>(element::i64, begin_shape));
    auto s_end = make_shared<op::v0::ShapeOf>(make_shared<op::v0::Parameter>(element::i64, end_shape));

    auto pad = this->make_op(arg, s_begin, s_end, op::PadMode::EDGE);

    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({{4, 7}, {2, 5}, {2, 6}}));
    EXPECT_THAT(get_shape_symbols(pad->get_output_partial_shape(0)), ElementsAre(nullptr, symbols[1], nullptr));
}

TYPED_TEST_P(PadTest, pad_preserve_partial_values_and_symbols_on_inputs) {
    auto arg_shape = PartialShape{1, {2, 5}, {1, 3}};
    auto symbols = set_shape_symbols(arg_shape);
    auto arg = make_shared<op::v0::Parameter>(element::i32, arg_shape);
    auto s = make_shared<op::v0::ShapeOf>(arg);

    auto pads_begin = make_shared<op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto pads_end = make_shared<op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{2});

    auto pad = this->make_op(s, pads_begin, pads_end, op::PadMode::EDGE);
    auto param = make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    auto bc = std::make_shared<op::v3::Broadcast>(param, pad, op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({1, 1, {2, 5}, {1, 3}, {1, 3}, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(bc->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[0], symbols[1], symbols[2], symbols[2], symbols[2]));
}

TYPED_TEST_P(PadTest, pad_begin_and_end_has_inf_interval_as_bounds) {
    auto arg_shape = PartialShape{9, {3, 5}, {3, 5}, {3, 4}, {3, 4}};
    auto begin_shape = PartialShape{-1, {0, 2}, -1, -1, {0, 1}};
    auto end_shape = PartialShape{-1, -1, {0, 2}, {0, 1}, -1};
    set_shape_symbols(arg_shape);
    set_shape_symbols(begin_shape);
    set_shape_symbols(end_shape);

    auto arg = make_shared<op::v0::Parameter>(element::f32, arg_shape);
    auto s_begin = make_shared<op::v0::ShapeOf>(make_shared<op::v0::Parameter>(element::i32, begin_shape));
    auto s_end = make_shared<op::v0::ShapeOf>(make_shared<op::v0::Parameter>(element::i32, end_shape));

    auto pad = this->make_op(arg, s_begin, s_end, op::PadMode::CONSTANT);

    EXPECT_EQ(pad->get_output_partial_shape(0), PartialShape({{9, -1}, {3, -1}, {3, -1}, {3, -1}, {3, -1}}));
    EXPECT_THAT(get_shape_symbols(pad->get_output_partial_shape(0)), Each(nullptr));
}

REGISTER_TYPED_TEST_SUITE_P(PadTest,
                            pad_default_ctor,
                            pad_arg_pad_value_type_mismatch,
                            pad_arg_pad_value_shape_not_compatible,
                            pad_pads_begin_shape_not_1D,
                            pad_pads_end_shape_not_1D,
                            pad_pads_begin_size_not_correct,
                            pad_pads_end_size_not_correct,
                            pad_arg_pads_begin_incompatible_type,
                            pad_arg_pads_end_incompatible_type,
                            pad_deduce_too_small_for_edge,
                            pad_deduce_too_small_for_reflect,
                            pad_pads_end_got_negative_value,
                            pad_pads_begin_got_negative_value,
                            pad_dynamic_output_with_dynamic_rank,
                            pad_dynamic_output_with_static_rank,
                            pad_any_dim_for_padding_reflect,
                            pad_any_dim_for_padding_edge,
                            pad_dynamic_input_type_with_static_value,
                            pad_preserve_partial_values_and_symbols_via_evaluates_bounds,
                            pad_begin_and_end_has_inf_interval_as_bounds,
                            pad_preserve_partial_values_and_symbols_on_inputs);

using PadOpTypes = Types<op::v1::Pad, op::v12::Pad>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, PadTest, PadOpTypes);
