// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "sequence_generator.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, split_v1_axis_const_positive) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f16, Shape{2, 3, 4});
    const auto axis = ov::op::v0::Constant::create(element::i64, {}, {1});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs,
                Each(AllOf(Property("Element type", &Output<Node>::get_element_type, element::f16),
                           Property("Shape", &Output<Node>::get_shape, Shape({2, 1, 4})))));
}

TEST(type_prop, split_v1_axis_const_negative) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 6});
    const auto axis = ov::op::v0::Constant::create(element::i64, {}, {-2});
    constexpr size_t num_splits = 2;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs,
                Each(AllOf(Property("Element type", &Output<Node>::get_element_type, element::i32),
                           Property("Shape", &Output<Node>::get_shape, Shape({1, 6})))));
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_known) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, 12, Dimension::dynamic()});
    const auto axis = ov::op::v0::Constant::create(element::i32, {}, {1});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape({2, 4, Dimension::dynamic()}))));
}

TEST(type_prop, split_v1_axis_const_only_data_axis_dim_known) {
    const auto data =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    const auto axis = ov::op::v0::Constant::create(element::i16, {}, {0});
    constexpr size_t num_splits = 2;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs,
                Each(Property(&Output<Node>::get_partial_shape,
                              PartialShape({1, Dimension::dynamic(), Dimension::dynamic()}))));
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_unknown) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{4, Dimension::dynamic(), 3, 5});
    const auto axis = ov::op::v0::Constant::create(element::i8, {}, {1});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs,
                Each(Property(&Output<Node>::get_partial_shape, PartialShape({4, Dimension::dynamic(), 3, 5}))));
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_interval_known_divisible) {
    const auto data =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{4, Dimension(3, 6), Dimension(3, 6), 5});
    const auto axis = ov::op::v0::Constant::create(element::i8, {}, {1});
    constexpr size_t num_splits = 2;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(
        outputs,
        Each(Property(&Output<Node>::get_partial_shape, PartialShape({4, Dimension(1, 3), Dimension(3, 6), 5}))));
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_interval_known_upper_bound_divisible) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{4, Dimension(2, 4), 3, 5});
    const auto axis = ov::op::v0::Constant::create(element::i8, {}, {1});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape({4, Dimension(0, 1), 3, 5}))));
}

TEST(type_prop, split_v1_axis_const_invalid_data_axis_dim_interval_known) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{4, Dimension(1, 2), 3, 5});
    const auto axis = ov::op::v0::Constant::create(element::i8, {}, {1});
    constexpr size_t num_splits = 3;

    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, num_splits),
                    NodeValidationFailure,
                    HasSubstr("The interval maximum of the dimension for data input shape along 'axis' must be "
                              "greater or equal to 'num_splits' attribute."));
}

TEST(type_prop, split_v1_axis_const_only_data_rank_known) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto axis = ov::op::v0::Constant::create(element::u64, {}, {1});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape::dynamic(4))));
}

TEST(type_prop, split_v1_axis_param_only_data_rank_known) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto axis = make_shared<ov::op::v0::Parameter>(element::u32, PartialShape{});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape::dynamic(4))));
}

TEST(type_prop, split_v1_axis_const_data_rank_unknown) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = ov::op::v0::Constant::create(element::u16, {}, {2});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape::dynamic())));
}

TEST(type_prop, split_v1_axis_param_data_rank_unknown) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<ov::op::v0::Parameter>(element::u8, PartialShape{});
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape::dynamic())));
}

TEST(type_prop, split_v1_axis_param_dynamic_ranks) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<ov::op::v0::Parameter>(element::u8, PartialShape::dynamic());
    constexpr size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs, Each(Property(&Output<Node>::get_partial_shape, PartialShape::dynamic())));
}

TEST(type_prop, split_v1_invalid_axis_et_f32) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 6});
    auto axis = ov::op::v0::Constant::create(element::f32, Shape{}, {1});

    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, 2),
                    NodeValidationFailure,
                    HasSubstr("Element type of 'axis' input must be integer."));
}

TEST(type_prop, split_v1_invalid_axis_et_boolean) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 6});
    auto axis = ov::op::v0::Constant::create(element::boolean, Shape{}, {1});

    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, 2),
                    NodeValidationFailure,
                    HasSubstr("Element type of 'axis' input must be integer."));
}

TEST(type_prop, split_v1_invalid_axis_not_a_scalar) {
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 6});
    auto axis = ov::op::v0::Constant::create(element::i64, Shape{2}, {0, 1});

    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, 1),
                    NodeValidationFailure,
                    HasSubstr("'axis' input must be a scalar."));
}

TEST(type_prop, split_v1_invalid_num_splits) {
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 6});
    auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    constexpr size_t num_splits = 0;

    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, num_splits),
                    ov::Exception,
                    HasSubstr("Attribute 'num_splits' must be greater than zero"));
}

TEST(type_prop, split_v1_invalid_axis_value) {
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 6});
    auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {-5});
    constexpr size_t num_splits = 4;

    // axis value not in the range [-2, 1]
    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, num_splits),
                    ov::Exception,
                    HasSubstr("Axis -5 out of the tensor rank range"));
}

TEST(type_prop, split_v1_incompatible_data_shape_with_num_splits) {
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 6});
    auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
    constexpr size_t num_splits = 4;

    OV_EXPECT_THROW(const auto split = make_shared<op::v1::Split>(data, axis, num_splits),
                    NodeValidationFailure,
                    HasSubstr("Dimension of data input shape along 'axis': 6 must be evenly "
                              "divisible by 'num_splits' attribute value: 4"));
}

using SplitTypePropTestParam = std::tuple<PartialShape,  // Input shape
                                          int64_t,       // Split axis
                                          size_t,        // Number of splits
                                          PartialShape   // Expected output(s) shape
                                          >;

class SplitTest : public TestWithParam<SplitTypePropTestParam> {
protected:
    void SetUp() override {
        std::tie(p_shape, axis, num_splits, exp_shape) = GetParam();
    }

    std::pair<ov::TensorSymbol, ov::TensorSymbol> make_in_exp_symbols() const {
        ov::TensorSymbol in_symbols;
        for (size_t i = 0; i < p_shape.size(); ++i)
            in_symbols.push_back(std::make_shared<Symbol>());

        auto exp_symbols = in_symbols;
        exp_symbols[axis] = nullptr;

        return {in_symbols, exp_symbols};
    }

    int64_t axis;
    size_t num_splits;
    PartialShape p_shape, exp_shape;
};

INSTANTIATE_TEST_SUITE_P(type_prop_static_shape,
                         SplitTest,
                         Values(
                             // Symbol is lost in this case see shape_infer for explanation.
                             std::make_tuple(PartialShape{6, 2}, 0, 1, PartialShape{6, 2}),
                             std::make_tuple(PartialShape{6}, 0, 2, PartialShape{3}),
                             std::make_tuple(PartialShape{3, 6, 7, 3}, 1, 2, PartialShape{3, 3, 7, 3})),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(type_prop_dynamic_shape,
                         SplitTest,
                         Values(
                             // Symbol is lost in this case see shape_infer for explanation.
                             std::make_tuple(PartialShape{Dimension(2, 6), 2}, 0, 1, PartialShape{Dimension(2, 6), 2}),
                             std::make_tuple(PartialShape{Dimension(4, 8)}, 0, 2, PartialShape{Dimension(2, 4)}),
                             std::make_tuple(PartialShape{3, 6, Dimension(10, 20), Dimension(1, 3)},
                                             2,
                                             2,
                                             PartialShape{3, 6, Dimension(5, 10), Dimension(1, 3)})),
                         PrintToStringParamName());

TEST_P(SplitTest, use_default_ctor) {
    constexpr auto dtype = element::f32;
    const auto param = make_shared<op::v0::Parameter>(dtype, p_shape);
    const auto axis_node = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, axis);

    const auto split = make_shared<op::v1::Split>();
    split->set_arguments(NodeVector{param, axis_node});
    split->set_num_splits(num_splits);
    split->validate_and_infer_types();

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(outputs,
                Each(AllOf(Property("Element type", &Output<Node>::get_element_type, dtype),
                           Property("Partial shape", &Output<Node>::get_partial_shape, exp_shape))));
}

TEST_P(SplitTest, symbols_propagation) {
    ov::TensorSymbol in_symbols, exp_symbols;
    std::tie(in_symbols, exp_symbols) = make_in_exp_symbols();

    set_shape_symbols(p_shape, in_symbols);
    const auto param = make_shared<op::v0::Parameter>(element::f32, p_shape);
    const auto axis_node = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, axis);
    const auto split = make_shared<op::v1::Split>(param, axis_node, num_splits);

    const auto outputs = split->outputs();
    EXPECT_EQ(outputs.size(), num_splits);
    EXPECT_THAT(
        outputs,
        Each(Property("Partial shape", &Output<Node>::get_partial_shape, ResultOf(get_shape_symbols, exp_symbols))));
}

using SplitBoundTestParam = std::tuple<PartialShape,              // Input shape
                                       size_t,                    // Number of splits
                                       std::vector<PartialShape>  // Expected splitted shapes
                                       >;

class SplitBoundTest : public TestWithParam<SplitBoundTestParam> {
protected:
    void SetUp() override {
        std::tie(p_shape, num_of_splits, exp_shapes) = GetParam();
    }

    std::pair<ov::TensorSymbol, std::vector<ov::TensorSymbol>> make_in_exp_symbols() const {
        ov::TensorSymbol in_symbols;
        for (size_t i = 0; i < p_shape.size(); ++i)
            in_symbols.push_back(std::make_shared<Symbol>());

        auto split_size = in_symbols.size() / num_of_splits;
        std::vector<ov::TensorSymbol> exp_symbols;
        for (auto it = in_symbols.begin(); it < in_symbols.end(); it += split_size) {
            exp_symbols.emplace_back(it, it + split_size);
        }

        return {in_symbols, exp_symbols};
    }

    ov::TensorSymbol in_symbols;
    std::vector<ov::TensorSymbol> out_symbols;

    PartialShape p_shape;
    size_t num_of_splits;
    std::vector<PartialShape> out_shapes, exp_shapes;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop_bounds_propagate,
    SplitBoundTest,
    Values(std::make_tuple(PartialShape{6, 2, 3, 4}, 2, std::vector<PartialShape>{{6, 2}, {3, 4}}),
           std::make_tuple(PartialShape{6, 2, 3, 4}, 4, std::vector<PartialShape>{{6}, {2}, {3}, {4}}),
           std::make_tuple(PartialShape{Dimension(2, 6), 2, 3, 4},
                           2,
                           std::vector<PartialShape>{{Dimension(2, 6), 2}, {3, 4}}),
           std::make_tuple(PartialShape{Dimension(2, 6), Dimension::dynamic(), Dimension(-1, 6), Dimension(7, -1)},
                           2,
                           std::vector<PartialShape>{{Dimension(2, 6), Dimension::dynamic()},
                                                     {Dimension(-1, 6), Dimension{7, -1}}})),
    PrintToStringParamName());

TEST_P(SplitBoundTest, propagate_symbol_and_dynamic_value) {
    const auto in_exp_symbols = make_in_exp_symbols();
    set_shape_symbols(p_shape, in_exp_symbols.first);

    constexpr auto et = element::i64;
    const auto symboled_param = std::make_shared<ov::op::v0::Parameter>(et, p_shape);
    const auto symboled_shape_of = std::make_shared<op::v0::ShapeOf>(symboled_param);

    const auto zero = std::vector<int64_t>{0};
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto split = std::make_shared<op::v1::Split>(symboled_shape_of, axis, num_of_splits);

    for (auto& output : split->outputs()) {
        const auto& bc = std::make_shared<op::v3::Broadcast>(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, PartialShape{1}),
            output);
        out_shapes.push_back(bc->get_output_partial_shape(0));
        out_symbols.push_back(get_shape_symbols(bc->get_output_partial_shape(0)));
    }

    EXPECT_EQ(out_shapes, exp_shapes);
    EXPECT_EQ(out_symbols, in_exp_symbols.second);
}
