// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gmock/gmock.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, variadic_split) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {1});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 4});
    const auto split = make_shared<op::v1::VariadicSplit>(data, axis, splits);
    EXPECT_EQ(split->outputs().size(), 2);
    EXPECT_EQ(split->get_output_shape(0), (Shape{2, 2}));
    EXPECT_EQ(split->get_output_shape(1), (Shape{2, 4}));
    EXPECT_EQ(split->get_output_element_type(0), element::i32);
    EXPECT_EQ(split->get_output_element_type(1), element::i32);

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{}, {-2}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{3}, {7, -1, 2}))
                  ->output(1)
                  .get_shape(),
              (Shape{3, 6}));

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{}, {-2}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{3}, {-1, 7, 2}))
                  ->output(0)
                  .get_shape(),
              (Shape{3, 6}));

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 1, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{1}, {2}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{3}, {3, 1, 2}))
                  ->output(2)
                  .get_shape(),
              (Shape{12, 1, 2}));

    EXPECT_EQ(make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, Shape{12, 6}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{1}, {1}),
                                                 op::Constant::create<int64_t>(element::i64, Shape{2}, {6, 0}))
                  ->output(1)
                  .get_shape(),
              (Shape{12, 0}));
}

TEST(type_prop, variadic_split_splits_rank) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {1});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{1, 2}, {2, 4});

    OV_EXPECT_THROW(const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis, splits),
                    NodeValidationFailure,
                    HasSubstr("Split lengths should be a 1-D tensor. Got 2 instead."));
}

TEST(type_prop, variadic_split_incorrect_sum) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {1});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 6});

    OV_EXPECT_THROW(const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis, splits),
                    NodeValidationFailure,
                    HasSubstr("Total length of splits: 7 must match the length of the chosen axis: 6"));
}

TEST(type_prop, variadic_split_incorrect_axis) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {-5});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 4});

    OV_EXPECT_THROW(const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis, splits),
                    ov::Exception,
                    HasSubstr("Parameter axis -5 out of the tensor rank range [-2, 1]."));
}

TEST(type_prop, variadic_split_splits_invalid_negative) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {1});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{2}, {-2, 4});

    OV_EXPECT_THROW(const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis, splits),
                    NodeValidationFailure,
                    HasSubstr("Invalid value -2 in split lengths input. Should be >= -1."));
}

TEST(type_prop, variadic_split_splits_multiple_negatives) {
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create<int64_t>(element::i64, Shape{}, {1});
    const auto splits = op::Constant::create<int64_t>(element::i64, Shape{3}, {-1, -1, 3});

    OV_EXPECT_THROW(const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis, splits),
                    NodeValidationFailure,
                    HasSubstr("Cannot infer split with multiple -1 values at 0 and 1"));
}

TEST(type_prop, variadic_split_shape_partially_dynamic) {
    // Variadic split shape {12,?} into {7,?}, {3,?} and {2,?}
    auto var_split1 =
        make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, PartialShape{12, Dimension()}),
                                           op::Constant::create<int64_t>(element::i64, Shape{}, {-2}),
                                           op::Constant::create<int64_t>(element::i64, Shape{3}, {7, -1, 2}));

    EXPECT_TRUE(var_split1->get_output_partial_shape(0).same_scheme(PartialShape{7, Dimension::dynamic()}));
    EXPECT_TRUE(var_split1->get_output_partial_shape(1).same_scheme(PartialShape{3, Dimension::dynamic()}));
    EXPECT_TRUE(var_split1->get_output_partial_shape(2).same_scheme(PartialShape{2, Dimension::dynamic()}));

    // Variadic split shape {?,?,6} into {?,?,3}, {?,?,1} and {?,?,2}
    auto var_split2 = make_shared<op::v1::VariadicSplit>(
        make_shared<op::Parameter>(element::i32, PartialShape{Dimension(), Dimension(), 6}),
        op::Constant::create<int64_t>(element::i64, Shape{}, {2}),
        op::Constant::create<int64_t>(element::i64, Shape{3}, {3, 1, 2}));

    EXPECT_TRUE(var_split2->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3}));
    EXPECT_TRUE(var_split2->get_output_partial_shape(1).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1}));
    EXPECT_TRUE(var_split2->get_output_partial_shape(2).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 2}));

    // Variadic split shape {?,6} into {?,6}, and {?,0}
    auto var_split3 =
        make_shared<op::v1::VariadicSplit>(make_shared<op::Parameter>(element::i32, PartialShape{Dimension(), 6}),
                                           op::Constant::create<int64_t>(element::i64, Shape{}, {1}),
                                           op::Constant::create<int64_t>(element::i64, Shape{2}, {6, 0}));

    EXPECT_TRUE(var_split3->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 6}));
    EXPECT_TRUE(var_split3->get_output_partial_shape(1).same_scheme(PartialShape{Dimension::dynamic(), 0}));
}

using VSplitTypePropTestParam = std::tuple<PartialShape,          // Input shape
                                           int64_t,               // Split axis
                                           std::vector<int64_t>,  // Split lengths
                                           PartialShapes          // Expected shapes
                                           >;

class VariadicSplitTest : public TestWithParam<VSplitTypePropTestParam> {
protected:
    void SetUp() override {
        std::tie(p_shape, axis, split_lengths, exp_shapes) = GetParam();
    }

    PartialShapes get_output_partial_shapes(const Node& n) {
        PartialShapes out;
        for (size_t i = 0; i < n.get_output_size(); ++i) {
            out.push_back(n.get_output_partial_shape(i));
        }

        return out;
    }

    int64_t axis;
    std::vector<int64_t> split_lengths;
    PartialShape p_shape;
    PartialShapes exp_shapes;
};

INSTANTIATE_TEST_SUITE_P(type_prop_static_shape,
                         VariadicSplitTest,
                         Values(std::make_tuple(PartialShape{6, 2}, 0, std::vector<int64_t>{6}, PartialShapes{{6, 2}}),
                                std::make_tuple(PartialShape{6, 2, 10},
                                                -1,
                                                std::vector<int64_t>{6, -1, 3},
                                                PartialShapes{{6, 2, 6}, {6, 2, 1}, {6, 2, 3}}),
                                std::make_tuple(PartialShape{1, 20, 3},
                                                1,
                                                std::vector<int64_t>{-1, 10, 3, 5},
                                                PartialShapes{{1, 2, 3}, {1, 10, 3}, {1, 3, 3}, {1, 5, 3}})),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    type_prop_dynamic_shape,
    VariadicSplitTest,
    Values(std::make_tuple(PartialShape{{2, 6}, 2}, 0, std::vector<int64_t>{4}, PartialShapes{{4, 2}}),
           std::make_tuple(PartialShape{{2, 6}, 2},
                           0,
                           std::vector<int64_t>{4, 1, -1},
                           PartialShapes{{4, 2}, {1, 2}, {-1, 2}}),
           std::make_tuple(PartialShape{{2, 4}, Dimension::dynamic()},
                           1,
                           std::vector<int64_t>{4, 1, -1, 3},
                           PartialShapes{{{2, 4}, 4}, {{2, 4}, 1}, {{2, 4}, -1}, {{2, 4}, 3}})),
    PrintToStringParamName());

TEST_P(VariadicSplitTest, dimension_propagation) {
    constexpr auto dtype = element::f32;
    const auto param = make_shared<op::Parameter>(dtype, p_shape);
    const auto axis_node = make_shared<op::Constant>(element::i32, Shape{}, axis);
    const auto lengths_node = std::make_shared<op::Constant>(element::i64, Shape{split_lengths.size()}, split_lengths);

    const auto var_split = make_shared<op::v1::VariadicSplit>();
    var_split->set_arguments(NodeVector{param, axis_node, lengths_node});
    var_split->validate_and_infer_types();

    EXPECT_EQ(var_split->get_output_size(), split_lengths.size());
    EXPECT_THAT(get_output_partial_shapes(*var_split), ElementsAreArray(exp_shapes));
}

TEST_P(VariadicSplitTest, use_default_ctor) {
    constexpr auto dtype = element::f32;
    const auto param = make_shared<op::Parameter>(dtype, p_shape);
    const auto axis_node = make_shared<op::Constant>(element::i32, Shape{}, axis);
    const auto lengths_node = std::make_shared<op::Constant>(element::i64, Shape{split_lengths.size()}, split_lengths);

    const auto var_split = make_shared<op::v1::VariadicSplit>();
    var_split->set_arguments(NodeVector{param, axis_node, lengths_node});
    var_split->validate_and_infer_types();

    EXPECT_EQ(var_split->get_output_size(), split_lengths.size());
    EXPECT_THAT(get_output_partial_shapes(*var_split), ElementsAreArray(exp_shapes));
}
