// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;
using namespace testing;

class AdaptiveAvgPoolV8Test : public TypePropOpTest<op::v8::AdaptiveAvgPool> {};

TEST_F(AdaptiveAvgPoolV8Test, default_ctor) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{2, 6, 3, 2});
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});

    const auto op = make_op();
    op->set_arguments(OutputVector{data, out_shape});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({2, 6, 5, 7}));
}

TEST_F(AdaptiveAvgPoolV8Test, static_dim_shape_prop) {
    auto data_shape = PartialShape{1, 6, 8, 9};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 6, 5, 7}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST_F(AdaptiveAvgPoolV8Test, dynamic_batch) {
    PartialShape data_shape{Dimension::dynamic(), 6, 8, 9};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({-1, 6, 5, 7}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST_F(AdaptiveAvgPoolV8Test, dynamic_channel) {
    PartialShape data_shape{1, Dimension::dynamic(), {10, 20}, 9};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, -1, 5, 7}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST_F(AdaptiveAvgPoolV8Test, dynamic_spatial) {
    PartialShape data_shape{1, 6, -1, -1};

    auto symbols = set_shape_symbols(data_shape);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 6, 5, 7}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST_F(AdaptiveAvgPoolV8Test, dynamic_output_shape) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9, 2});
    auto out_shape = make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 6, -1, -1, -1}));
}

TEST_F(AdaptiveAvgPoolV8Test, output_shape_as_parameter) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9, 2});
    auto out_shape = make_shared<Parameter>(element::i64, PartialShape{3});
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({1, 6, -1, -1, -1}));
}

TEST_F(AdaptiveAvgPoolV8Test, data_dynamic_rank) {
    auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto out_shape = make_shared<Parameter>(element::i32, Shape{3});
    const auto op = make_op(data, out_shape);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(AdaptiveAvgPoolV8Test, preserve_partial_values_and_symbols_on_output_shape_input) {
    auto data_shape = PartialShape{{1, 2}, {2, 4}, 5, {10, 20}, -1};
    auto d_symbols = set_shape_symbols(data_shape);
    auto out_shape = PartialShape{{2, 6}, 3, {12, 13}};
    auto o_symbols = set_shape_symbols(out_shape);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto spatial_dim_shape = make_shared<ShapeOf>(make_shared<Parameter>(element::i64, out_shape));
    const auto op = make_op(data, spatial_dim_shape);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{1, 2}, {2, 4}, {2, 6}, 3, {12, 13}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(d_symbols[0], d_symbols[1], o_symbols[0], o_symbols[1], o_symbols[2]));
}

TEST_F(AdaptiveAvgPoolV8Test, out_spatial_shape_size_not_match_data_spatial_dimensions) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{2, 3, 5, 6});
    auto out_shape = make_shared<Parameter>(element::i32, Shape{3});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape),
                    NodeValidationFailure,
                    HasSubstr("Output shape for spatial dimension not compatible with data shape."));
}

TEST_F(AdaptiveAvgPoolV8Test, unsupported_input_shape) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6});
    auto out_shape = Constant::create<int64_t>(element::i64, Shape{}, {1});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D, 4D or 5D tensor for the input. Got:"));
}

TEST_F(AdaptiveAvgPoolV8Test, wrong_out_shape) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9});
    auto out_shape = Constant::create<int64_t>(element::i64, Shape{3}, {5, 7, 8});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape),
                    NodeValidationFailure,
                    HasSubstr("Output shape for spatial dimension not compatible with data shape."));
}
