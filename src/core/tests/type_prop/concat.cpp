// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/graph_rewrite.hpp"

using namespace std;
using namespace testing;

TEST(type_prop, concat_deduce) {
    // Deduce type
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 4});
    auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
    EXPECT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 12, 4}));
}

TEST(type_prop, concat_deduce_wrong_rank) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                     ov::Shape{
                                                         2,
                                                         2,
                                                     });
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_shape) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 5});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_oob) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 5});

    OV_EXPECT_THROW(ignore = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 3),
                    ov::AssertFailure,
                    HasSubstr("Axis 3 out of the tensor rank range"));
}

TEST(type_prop, concat_deduce_axis_barely_in_bounds) {
    // Deduce type
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 8});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 12});
    auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 2);
    EXPECT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 3, 24}));
}

TEST(type_prop, concat_deduce_elem_type_mismatch) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 4});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_et_consistent) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 4});
    auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);

    EXPECT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 12, 4}));
}

TEST(type_prop, concat_partial_et_inconsistent) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::dynamic, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 2, 4});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent element types not detected (some dynamic)";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_rank_inconsistent) {
    auto param0 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, ov::Dimension::dynamic(), 3});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto param2 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, ov::Dimension::dynamic(), 4});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent ranks not detected (some args rank-dynamic, some args rank-static "
                  "dynamic)";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_inconsistent) {
    auto param0 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, ov::Dimension::dynamic(), 3});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto param2 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 3, ov::Dimension::dynamic()});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_intransitively_inconsistent) {
    auto param0 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, ov::Dimension::dynamic(), 3});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto param2 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                           ov::PartialShape{ov::Dimension::dynamic(), 3, ov::Dimension::dynamic()});
    auto param3 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 3, ov::Dimension::dynamic()});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2, param3}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_with_concat_axis_static_dims_inconsistent) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2, 3});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto param2 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 3, ov::Dimension::dynamic()});

    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_compatible_result_static) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2, 3});
    auto param1 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 4, 3});
    auto param2 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, ov::Dimension::dynamic()});
    auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);

    ASSERT_EQ(c->get_shape(), (ov::Shape{2, 9, 3}));
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_dims_incompatible) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2, 3});
    auto param1 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 4, 3});
    auto param2 =
        make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 3, ov::Dimension::dynamic()});
    try {
        auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                                         "have equal dimension everywhere except on the concatenation axis"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_negative_axis_correct) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{7, 2, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 4});

    auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, -3);

    EXPECT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{12, 2, 4}));
}

TEST(type_prop, concat_partial_negative_axis_incorrect) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 7, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 4});

    OV_EXPECT_THROW(ignore = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, -4),
                    ov::AssertFailure,
                    HasSubstr("Axis -4 out of the tensor rank range"));
}

/** \brief Test uses evaluate lower/upper and symbol of concat op. */
TEST(type_prop, concat_dynamic_value_and_symbol_propagation) {
    ov::Dimension marked_0 = ov::Dimension(3);
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(A);
    ov::PartialShape target_0 = ov::PartialShape{marked_0, 4};

    ov::Dimension marked_1 = ov::Dimension(5);
    marked_1.set_symbol(B);
    ov::PartialShape target_1 = ov::PartialShape{4, marked_1, 9};

    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto param_0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, target_0);
    auto shape_0 = make_shared<ov::op::v0::ShapeOf>(param_0);

    auto param_1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, target_1);
    auto shape_1 = make_shared<ov::op::v0::ShapeOf>(param_1);

    auto five = ov::op::v0::Constant::create(ov::element::i64, {1}, {5});
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{shape_0, five, shape_1}, 0);

    auto bc = make_shared<ov::op::v1::Broadcast>(param, target_shape);
    EXPECT_EQ(bc->get_shape(), (ov::Shape{3, 4, 5, 4, 5, 9}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    const auto symbols = get_shape_symbols(output_shape);
    ASSERT_THAT(symbols, ElementsAre(A, nullptr, nullptr, nullptr, B, nullptr));
}

/** \brief Test uses evaluate lower/upper and symbol of concat op. */
TEST(type_prop, concat_dynamic_value_and_symbol_propagation_1) {
    ov::Dimension marked_0 = ov::Dimension(3);
    auto A = std::make_shared<ov::Symbol>(), B = std::make_shared<ov::Symbol>();
    marked_0.set_symbol(A);
    ov::PartialShape target_0 = ov::PartialShape{marked_0, 4};

    ov::Dimension marked_1 = ov::Dimension(5);
    marked_1.set_symbol(B);
    ov::PartialShape target_1 = ov::PartialShape{4, marked_1, 9};

    auto param = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto param_0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, target_0);
    auto shape_0 = make_shared<ov::op::v0::ShapeOf>(param_0);
    auto convert_0 = make_shared<ov::op::v0::Convert>(shape_0, ov::element::i8);

    auto param_1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, target_1);
    auto shape_1 = make_shared<ov::op::v0::ShapeOf>(param_1);
    auto convert_1 = make_shared<ov::op::v0::Convert>(shape_1, ov::element::i8);

    auto five = ov::op::v0::Constant::create(ov::element::i8, {1}, {5});
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{convert_0, five, convert_1}, 0);

    auto convert = std::make_shared<ov::op::v0::Convert>(target_shape, ov::element::i64);

    auto bc = make_shared<ov::op::v1::Broadcast>(param, target_shape);
    EXPECT_EQ(bc->get_shape(), (ov::Shape{3, 4, 5, 4, 5, 9}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    const auto symbols = get_shape_symbols(output_shape);
    ASSERT_THAT(symbols, ElementsAre(A, nullptr, nullptr, nullptr, B, nullptr));
}

TEST(type_prop, concat_interval_dimensions) {
    auto param0 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 4});
    auto param1 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{7, 2, 4});
    auto param2 = make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 4});

    auto c = make_shared<ov::op::v0::Concat>(ov::NodeVector{param0, param1, param2}, -3);

    EXPECT_EQ(c->get_element_type(), ov::element::f32);
    ASSERT_EQ(c->get_shape(), (ov::Shape{12, 2, 4}));
}

using PartialShapeVector = std::vector<ov::PartialShape>;
using ConcatTestParams = std::tuple<PartialShapeVector,          // input shapes
                                    std::tuple<int64_t,          // concatenation axis
                                               ov::PartialShape  // expected shape
                                               >>;

class ConcatTest : public TestWithParam<ConcatTestParams> {
protected:
    void SetUp() override {
        int64_t axis;
        PartialShapeVector input_shapes;
        ov::pass::NodeRegistry params;

        std::forward_as_tuple(input_shapes, std::tie(axis, exp_shape)) = GetParam();

        for (const auto& shape : input_shapes) {
            params.make<ov::op::v0::Parameter>(ov::element::f32, shape);
        }

        c = make_shared<ov::op::v0::Concat>(params.get(), axis);
    }

    ov::PartialShape exp_shape;
    std::shared_ptr<ov::op::v0::Concat> c;
};

const auto shapes_with_interval_dim = Values(PartialShapeVector{(ov::PartialShape::dynamic()),
                                                                {2, ov::Dimension(2, 5), 3, 1},
                                                                {2, 4, 3, ov::Dimension(1, 4)},
                                                                {2, 4, 3, 1}});

INSTANTIATE_TEST_SUITE_P(
    type_prop_interval_dim_mixed_ranks,
    ConcatTest,
    Combine(shapes_with_interval_dim,
            Values(std::make_tuple(1, ov::PartialShape({2, ov::Dimension(10, -1), 3, 1})),  // axis 1
                   std::make_tuple(-1, ov::PartialShape({2, 4, 3, ov::Dimension(3, -1)})),  // axis 2
                   std::make_tuple(2, ov::PartialShape({2, 4, ov::Dimension(9, -1), 1}))    // axis 3
                   )),
    PrintToStringParamName());

const auto shapes_all_dynamic_ranks = Values(PartialShapeVector{(ov::PartialShape::dynamic()),
                                                                (ov::PartialShape::dynamic()),
                                                                (ov::PartialShape::dynamic()),
                                                                (ov::PartialShape::dynamic())});

INSTANTIATE_TEST_SUITE_P(type_prop_dynamic_ranks_against_axis_range,
                         ConcatTest,
                         Combine(shapes_all_dynamic_ranks,
                                 Combine(Range<int64_t>(-4, 4), Values(ov::PartialShape::dynamic()))),
                         PrintToStringParamName());

const auto shapes_static_dynamic_ranks =
    Values(PartialShapeVector{ov::PartialShape({4, 2, ov::Dimension::dynamic(), 3}),
                              ov::PartialShape::dynamic(),
                              ov::PartialShape({4, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic()})});

INSTANTIATE_TEST_SUITE_P(
    type_prop_mixed_ranks_and_dims,
    ConcatTest,
    Combine(shapes_static_dynamic_ranks,
            Values(
                // concat all dynamic dims
                std::make_tuple(2, ov::PartialShape({4, 2, ov::Dimension::dynamic(), 3})),
                // concat dynamic and interval dim
                std::make_tuple(1, ov::PartialShape({4, ov::Dimension(4, -1), ov::Dimension::dynamic(), 3})))),
    PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    type_prop_1d_shapes,
    ConcatTest,
    Values(
        // concat all dynamic dims
        std::make_tuple(PartialShapeVector{{-1}, {-1}, {-1}}, std::make_tuple(0, ov::PartialShape({-1}))),
        // concat dynamic and not matching static dims
        std::make_tuple(PartialShapeVector{{3}, ov::PartialShape::dynamic(), {2}},
                        std::make_tuple(0, ov::PartialShape({ov::Dimension(5, -1)}))),
        // concat all static dim
        std::make_tuple(PartialShapeVector{{3}, {3}, {3}}, std::make_tuple(0, ov::PartialShape({9}))),
        // concat dynamic and interval dim
        std::make_tuple(PartialShapeVector{{3}, {ov::Dimension::dynamic()}, {ov::Dimension(3, 4)}},
                        std::make_tuple(0, ov::PartialShape({ov::Dimension(6, -1)})))),
    PrintToStringParamName());

/** \brief ov::Shape propagation no exception. */
TEST_P(ConcatTest, partial_shape_propagation) {
    ASSERT_EQ(c->get_default_output().get_partial_shape(), exp_shape);
}
