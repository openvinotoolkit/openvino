// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "sequence_generator.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, unsqueeze) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);

    EXPECT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_EQ(unsqueeze->get_shape(), (Shape{4, 1, 1, 1, 4, 1, 8}));
}

TEST(type_prop, unsqueeze_incorrect_axes_shape) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{1, 1, 1}, 1);

    try {
        auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
        FAIL() << "Unsqueeze axes invalid rank not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Second input (axes) should not be of rank higher than 1");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unsqueeze_positive_axis_gt_ouput_rank) {
    constexpr int64_t bad_axis = 6;
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{1}, bad_axis);

    try {
        auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
        FAIL() << "Unsqueeze axes invalid rank not detected";
    } catch (const NodeValidationFailure& error) {
        const auto exp_msg = "Axis " + std::to_string(bad_axis) + " out of the tensor rank range";
        EXPECT_HAS_SUBSTRING(error.what(), exp_msg);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unsqueeze_negative_axis_gt_ouput_rank) {
    constexpr int64_t bad_axis = -7;
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{1}, bad_axis);

    try {
        auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
        FAIL() << "Unsqueeze axes invalid rank not detected";
    } catch (const NodeValidationFailure& error) {
        const auto exp_msg = "Axis " + std::to_string(bad_axis) + " out of the tensor rank range";
        EXPECT_HAS_SUBSTRING(error.what(), exp_msg);
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, unsqueeze_empty_axes) {
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node = make_shared<ov::op::v0::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    try {
        auto unsqueeze = make_shared<op::v0::Unsqueeze>(param, axes_node);
        FAIL() << "Unsqueeze axes empty not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "'axes' input is mandatory");
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

using UnSqueezeTypePropTestParam = std::tuple<PartialShape,          // Input shape
                                              std::vector<int64_t>,  // Unsqueeze axis
                                              PartialShape           // Expected shape
                                              >;

class UnsqueezeTest : public WithParamInterface<UnSqueezeTypePropTestParam>, public UnSqueezeFixture {
protected:
    void SetUp() override {
        std::tie(p_shape, axes, exp_shape) = GetParam();
        UnSqueezeFixture::SetUp();
    }

    std::pair<ov::TensorSymbol, ov::TensorSymbol> make_in_exp_symbols() const {
        ov::TensorSymbol in_symbols;
        for (size_t i = 0; i < p_shape.size(); ++i)
            in_symbols.push_back(std::make_shared<Symbol>());

        auto unique_axes = std::set<int64_t>(axes.begin(), axes.end());
        auto out_rank = unique_axes.size() + p_shape.size();

        std::set<int64_t> no_symbol_axes;
        for (const auto& axis : axes) {
            no_symbol_axes.insert(axis < 0 ? axis + out_rank : axis);
        }

        auto exp_symbols = in_symbols;
        for (const auto& axis : no_symbol_axes) {
            if (axis < static_cast<int64_t>(exp_symbols.size())) {
                exp_symbols.insert(exp_symbols.begin() + axis, nullptr);
            } else {
                exp_symbols.push_back(nullptr);
            }
        }
        return {in_symbols, exp_symbols};
    }

    std::vector<int64_t> axes;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop_expand_dynamic_shape,
    UnsqueezeTest,
    Values(std::make_tuple(PartialShape::dynamic(), std::vector<int64_t>{-1, -5}, PartialShape::dynamic()),
           std::make_tuple(PartialShape::dynamic(3),
                           std::vector<int64_t>{-1, -5},
                           PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 1}),
           std::make_tuple(PartialShape::dynamic(5),
                           std::vector<int64_t>{1, 2},
                           PartialShape{Dimension::dynamic(),
                                        1,
                                        1,
                                        Dimension::dynamic(),
                                        Dimension::dynamic(),
                                        Dimension::dynamic(),
                                        Dimension::dynamic()}),
           std::make_tuple(PartialShape{Dimension(3, 5), Dimension(-1, 2)},
                           std::vector<int64_t>{0},
                           PartialShape{1, Dimension(3, 5), Dimension(-1, 2)}),
           std::make_tuple(PartialShape{2, Dimension::dynamic(), 4, Dimension(2, 7), 6},
                           std::vector<int64_t>{3, 1, 5},
                           PartialShape{2, 1, Dimension::dynamic(), 1, 4, 1, Dimension(2, 7), 6})),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    type_prop_expand_shape_at_edges,
    UnsqueezeTest,
    Values(std::make_tuple(PartialShape{}, std::vector<int64_t>{0}, PartialShape{1}),
           std::make_tuple(PartialShape{-1}, std::vector<int64_t>{-1}, PartialShape{-1, 1}),
           std::make_tuple(PartialShape{0}, std::vector<int64_t>{-1}, PartialShape{0, 1}),
           std::make_tuple(PartialShape{5}, std::vector<int64_t>{0}, PartialShape{1, 5}),
           std::make_tuple(PartialShape{5}, std::vector<int64_t>{-2}, PartialShape{1, 5}),
           std::make_tuple(PartialShape{5}, std::vector<int64_t>{1}, PartialShape{5, 1}),
           std::make_tuple(PartialShape{5}, std::vector<int64_t>{-1}, PartialShape{5, 1}),
           std::make_tuple(PartialShape{2, 3}, std::vector<int64_t>{0}, PartialShape{1, 2, 3}),
           std::make_tuple(PartialShape{2, 3, 6}, std::vector<int64_t>{0, -1}, PartialShape{1, 2, 3, 6, 1}),
           std::make_tuple(PartialShape{2, 3, 6}, std::vector<int64_t>{-1, -5}, PartialShape{1, 2, 3, 6, 1})),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    type_prop_expand_shape_inside,
    UnsqueezeTest,
    Values(
        std::make_tuple(PartialShape{5}, std::vector<int64_t>{1, -1}, PartialShape{5, 1, 1}),
        std::make_tuple(PartialShape{5, 4}, std::vector<int64_t>{1}, PartialShape{5, 1, 4}),
        std::make_tuple(PartialShape{8, 2}, std::vector<int64_t>{-2}, PartialShape{8, 1, 2}),
        std::make_tuple(PartialShape{3, 4}, std::vector<int64_t>{1, -2}, PartialShape{3, 1, 1, 4}),
        std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{1, 3}, PartialShape{2, 1, 3, 1, 4}),
        std::make_tuple(PartialShape{3, 6, 7, 3}, std::vector<int64_t>{-2, 3, 1}, PartialShape{3, 1, 6, 1, 7, 1, 3})),
    PrintToStringParamName());

// These are cases with repeated axis which should throw exception as numpy.
// Current implementation allow this in shape inference by removing duplicates.
INSTANTIATE_TEST_SUITE_P(
    type_prop_repeated_axis,
    UnsqueezeTest,
    Values(std::make_tuple(PartialShape{5}, std::vector<int64_t>{1, 1}, PartialShape{5, 1}),
           std::make_tuple(PartialShape{5}, std::vector<int64_t>{-1, -1}, PartialShape{5, 1}),
           std::make_tuple(PartialShape{5}, std::vector<int64_t>{1, -1, 1}, PartialShape{5, 1, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{1, 1}, PartialShape{2, 1, 3, 4}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{1, 1, 3, 3}, PartialShape{2, 1, 3, 1, 4}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{1, -1, 1}, PartialShape{2, 1, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{1, -1}, PartialShape{2, 1, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{-1, -1}, PartialShape{2, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{-1, -1, 1}, PartialShape{2, 1, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{-1, 4, -1}, PartialShape{2, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{-1, 4}, PartialShape{2, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4}, std::vector<int64_t>{4, -1}, PartialShape{2, 3, 4, 1}),
           std::make_tuple(PartialShape{2, 3, 4},
                           std::vector<int64_t>{4, 4, -2, -2, -1, -1},
                           PartialShape{2, 3, 4, 1, 1})),
    PrintToStringParamName());

TEST_P(UnsqueezeTest, dimension_propagation_const_axis_i8) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i8, Shape{axes.size()}, axes);
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(param, axes_node);

    EXPECT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_EQ(unsqueeze->get_output_partial_shape(0), exp_shape);
}

TEST_P(UnsqueezeTest, dimension_propagation_const_axis_i32) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{axes.size()}, axes);
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(param, axes_node);

    EXPECT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_EQ(unsqueeze->get_output_partial_shape(0), exp_shape);
}

TEST_P(UnsqueezeTest, dimension_propagation_dynamic_axis_shape) {
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(param, axes_node);

    EXPECT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_EQ(unsqueeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_P(UnsqueezeTest, dimension_propagation_static_rank_dynamic_dim_axis_shape) {
    const auto axes_node = std::make_shared<ov::op::v0::Parameter>(element::u32, PartialShape{Dimension(2, 6)});
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(param, axes_node);

    EXPECT_EQ(unsqueeze->get_element_type(), element::f32);
    EXPECT_EQ(unsqueeze->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_P(UnsqueezeTest, use_default_ctor) {
    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{axes.size()}, axes);

    const auto unsqueeze = make_shared<op::v0::Unsqueeze>();
    unsqueeze->set_arguments(ov::NodeVector{param, axes_node});
    unsqueeze->validate_and_infer_types();

    EXPECT_EQ(unsqueeze->get_output_element_type(0), element::f32);
    EXPECT_EQ(unsqueeze->get_output_partial_shape(0), exp_shape);
}

TEST_P(UnsqueezeTest, symbols_propagation) {
    if (p_shape.rank().is_dynamic()) {
        GTEST_SKIP() << "No dimension to set symbol";
    }
    ov::TensorSymbol in_symbols, exp_symbols;
    std::tie(in_symbols, exp_symbols) = make_in_exp_symbols();

    set_shape_symbols(p_shape, in_symbols);
    param = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);

    const auto axes_node = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{axes.size()}, axes);
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(param, axes_node);

    EXPECT_EQ(get_shape_symbols(unsqueeze->get_output_partial_shape(0)), exp_symbols);
}

using UnsqueezeBoundTest = UnSqueezeBoundTest;

INSTANTIATE_TEST_SUITE_P(
    type_prop_bounds_propagate,
    UnsqueezeBoundTest,
    Values(std::make_tuple(PartialShape::dynamic(2), PartialShape::dynamic(1)),
           std::make_tuple(PartialShape{Dimension(-1)}, PartialShape{Dimension(-1)}),
           std::make_tuple(PartialShape{Dimension::dynamic(), 8}, PartialShape{Dimension::dynamic()}),
           std::make_tuple(PartialShape{Dimension(2, 5), Dimension::dynamic()}, PartialShape{Dimension(2, 5)}),
           std::make_tuple(PartialShape{Dimension(2, -1), Dimension::dynamic()}, PartialShape{{2, -1}}),
           std::make_tuple(PartialShape{Dimension(-1, 3), Dimension::dynamic()}, PartialShape{Dimension(-1, 3)}),
           std::make_tuple(PartialShape{5}, PartialShape{5}),
           std::make_tuple(PartialShape{2, 6}, PartialShape{2})),
    PrintToStringParamName());

/**
 * \brief Check symbol and dynamic value propagation.
 *
 * Test use evaluate symbol, lower/upper.
 */
TEST_P(UnsqueezeBoundTest, propagate_symbol_and_dynamic_value) {
    PartialShape symboled_shape = PartialShape{p_shape};
    for (size_t s = 0; s < symboled_shape.size(); ++s)
        in_symbols.push_back(std::make_shared<Symbol>());
    set_shape_symbols(symboled_shape, in_symbols);

    constexpr auto et = element::i64;
    const auto symboled_param = std::make_shared<ov::op::v0::Parameter>(et, symboled_shape);
    const auto symboled_shape_of = std::make_shared<op::v0::ShapeOf>(symboled_param);

    const auto zero = std::vector<int64_t>{0};
    const auto axis = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto indices = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto gather = std::make_shared<op::v7::Gather>(symboled_shape_of, indices, axis);
    const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(gather, axis);

    const auto bc = std::make_shared<op::v3::Broadcast>(param, unsqueeze);

    EXPECT_EQ(bc->get_output_partial_shape(0), exp_shape);
    const auto symbols = get_shape_symbols(bc->get_output_partial_shape(0));
    EXPECT_THAT(symbols, ElementsAre(in_symbols.front()));
}
