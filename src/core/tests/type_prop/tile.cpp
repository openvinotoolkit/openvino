// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "dimension_tracker.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

class TypePropTileTest : public TypePropOpTest<op::v0::Tile> {
protected:
    PartialShape shape_in;
};

TEST_F(TypePropTileTest, exception_if_repeats_are_float) {
    const auto data = make_shared<op::Parameter>(element::f64, Shape{2, 3, 4});
    const auto repeats = op::Constant::create(element::f32, Shape{3}, {3, 2, 1});

    OV_EXPECT_THROW(auto op = make_op(data, repeats),
                    NodeValidationFailure,
                    HasSubstr("Tile repeats must have any integer element type, but has"));
}

TEST_F(TypePropTileTest, exception_if_repeats_shape_is_not_rank_1) {
    const auto data = make_shared<op::Parameter>(element::f64, Shape{2, 3, 4});
    const auto repeats = op::Constant::create(element::i16, Shape{3, 1}, {3, 2, 1});

    OV_EXPECT_THROW(auto op = make_op(data, repeats),
                    NodeValidationFailure,
                    HasSubstr("Tile repeats must be of rank 1"));
}

TEST_F(TypePropTileTest, repeats_has_negative_values) {
    const auto data = make_shared<op::Parameter>(element::i32, PartialShape{-1, 3, 4, {-1, 5}, {4, -1}});
    const auto repeats = op::Constant::create(element::i8, Shape{5}, {-1, -2, 1, -1, -1});
    auto op = make_op(data, repeats);

    EXPECT_EQ(op->get_element_type(), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({0, 0, 4, 0, 0}));
}

TEST_F(TypePropTileTest, repeats_are_undefined_and_its_rank_lt_data_rank) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    const auto repeats = make_shared<op::Parameter>(element::i32, Shape{2});

    const auto op = make_op(data, repeats);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST_F(TypePropTileTest, repeats_are_undefined_and_its_rank_gt_data_rank) {
    const auto data = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    const auto repeats = make_shared<op::Parameter>(element::i32, Shape{5});

    const auto op = make_op(data, repeats);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(5));
}

TEST_F(TypePropTileTest, data_dynamic_rank_repeats_are_undefined) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto repeats = make_shared<op::Parameter>(element::i32, Shape{5});

    const auto op = make_op(data, repeats);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropTileTest, data_and_repeats_are_dynamic_rank) {
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto repeats = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(data, repeats);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

using TileTestParam = std::tuple<PartialShape, std::vector<int64_t>, PartialShape>;

class TileTest : public TypePropTileTest, public WithParamInterface<TileTestParam> {
protected:
    void SetUp() override {
        std::tie(shape_in, repeats_val, exp_shape) = GetParam();
    }

    std::vector<size_t> get_exp_labels() const {
        auto labels = get_shape_labels(shape_in);

        if (!labels.empty()) {
            auto repeats = repeats_val;
            int64_t size_diff = labels.size() - repeats.size();

            if (size_diff >= 0) {
                repeats.insert(repeats.begin(), size_diff, 1);
            } else {
                labels.insert(labels.begin(), -size_diff, ov::no_label);
            }

            std::transform(labels.begin(),
                           labels.end(),
                           repeats.begin(),
                           labels.begin(),
                           [](const size_t label, const int64_t repeat) {
                               return (label != ov::no_label && repeat == 1) ? label : ov::no_label;
                           });
        }
        return labels;
    }

    PartialShape exp_shape;
    std::vector<int64_t> repeats_val;
};

INSTANTIATE_TEST_SUITE_P(
    type_prop_static_shape,
    TileTest,
    Values(
        std::make_tuple(PartialShape{1, 2}, std::vector<int64_t>{0, 0}, PartialShape{0, 0}),
        std::make_tuple(PartialShape{3, 7, 1, 2, 4}, std::vector<int64_t>{2, 1, 7, 1, 2}, PartialShape{6, 7, 7, 2, 8}),
        std::make_tuple(PartialShape{1, 4, 2}, std::vector<int64_t>(3, 1), PartialShape{1, 4, 2}),
        std::make_tuple(PartialShape{1, 2, 4}, std::vector<int64_t>{2, 1}, PartialShape{1, 4, 4}),
        std::make_tuple(PartialShape{3, 6, 7, 1, 2, 4}, std::vector<int64_t>{2, 2}, PartialShape{3, 6, 7, 1, 4, 8}),
        std::make_tuple(PartialShape{1, 2, 4}, std::vector<int64_t>{2, 1, 1, 1}, PartialShape{2, 1, 2, 4}),
        std::make_tuple(PartialShape{1, 2, 4}, std::vector<int64_t>{2, 1, 2, 3, 4}, PartialShape{2, 1, 2, 6, 16})),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    type_prop_dynamic_shape,
    TileTest,
    Values(
        std::make_tuple(PartialShape{{1, 5}, {2, -1}}, std::vector<int64_t>{0, 0}, PartialShape{0, 0}),
        std::make_tuple(PartialShape{{3, -1}, {-1, 7}, -1, {2, 3}, {-1, 2}},
                        std::vector<int64_t>{2, 1, 7, 1, 2},
                        PartialShape{{6, -1}, {-1, 7}, -1, {2, 3}, {-1, 4}}),
        std::make_tuple(PartialShape{1, 4, 2}, std::vector<int64_t>(3, 1), PartialShape{1, 4, 2}),
        std::make_tuple(PartialShape{3, 6, {7, 9}, 1, {2, 8}, 4},
                        std::vector<int64_t>{2, 2},
                        PartialShape{3, 6, {7, 9}, 1, {4, 16}, 8}),
        std::make_tuple(PartialShape{-1, -1, -1}, std::vector<int64_t>{2, 1, 2, 3, 4}, PartialShape{2, 1, -1, -1, -1})),
    PrintToStringParamName());

TEST_P(TileTest, default_ctor) {
    constexpr auto dt = element::f16;
    const auto data = make_shared<op::Parameter>(dt, shape_in);
    const auto repeats = op::Constant::create(element::i64, Shape{repeats_val.size()}, repeats_val);

    const auto op = make_op();
    op->set_arguments(OutputVector{data, repeats});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), dt);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}

TEST_P(TileTest, propagate_shapes_and_labels) {
    ASSERT_TRUE(shape_in.rank().is_static()) << "Cannot test labels propagation for dynamic rank.";

    constexpr auto dt = element::f32;
    const auto data = make_shared<op::Parameter>(dt, shape_in);
    const auto repeats = op::Constant::create(element::i64, Shape{repeats_val.size()}, repeats_val);

    const auto op = make_op(data, repeats);

    EXPECT_EQ(op->get_element_type(), dt);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
    EXPECT_EQ(get_shape_labels(op->get_output_partial_shape(0)), get_exp_labels());
}
