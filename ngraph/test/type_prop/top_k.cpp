// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

// Since v3::TopK is backward compatible with v1::TopK all of these tests should pass
template <typename T>
class topk_type_prop : public ::testing::Test
{
};
TYPED_TEST_CASE_P(topk_type_prop);

TYPED_TEST_P(topk_type_prop, topk_negative_axis_support)
{
    const auto data_shape = Shape{1, 2, 3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto k = op::Constant::create(element::i64, Shape{}, {2});
    const int64_t axis = -2;

    const auto topk = make_shared<TypeParam>(data, k, axis, "max", "value");

    ASSERT_EQ(topk->get_provided_axis(), axis);
    const auto expect_shape = Shape{1, 2, 2, 4};
    ASSERT_EQ(topk->get_output_shape(0), expect_shape);
    ASSERT_EQ(topk->get_output_shape(1), expect_shape);
}

TYPED_TEST_P(topk_type_prop, topk_default_index_element_type)
{
    const auto data_shape = Shape{1, 2, 3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto k = op::Constant::create(element::i64, Shape{}, {2});
    const int64_t axis = -2;

    const auto op = make_shared<op::v1::TopK>(data, k, axis, "max", "value");
    ASSERT_EQ(op->get_index_element_type(), element::i32);
}

TYPED_TEST_P(topk_type_prop, topk_negative_axis_dynamic_rank)
{
    const auto data_shape = PartialShape::dynamic();
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto k = op::Constant::create(element::i64, Shape{}, {2});
    const int64_t axis = -2;
    const auto topk = make_shared<TypeParam>(data, k, axis, "max", "value");

    try
    {
        topk->get_axis();
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Normalized axis of TopK is unknown"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TYPED_TEST_P(topk_type_prop, topk_v1_partial_ouptut)
{
    auto data_shape = PartialShape{2, 10};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    {
        auto k = make_shared<op::Parameter>(element::i32, PartialShape({}));
        auto topk = make_shared<TypeParam>(data, k, 1, "max", "value");
        EXPECT_EQ(topk->get_output_partial_shape(0), PartialShape({2, Dimension(0, 10)}));
    }
    {
        auto k = make_shared<op::Constant>(element::i32, Shape{}, 3);
        auto topk = make_shared<TypeParam>(data, k, 1, "max", "value");
        EXPECT_EQ(topk->get_output_shape(0), Shape({2, 3}));
        EXPECT_EQ(topk->get_output_partial_shape(0), PartialShape({2, 3}));
    }
}

TYPED_TEST_P(topk_type_prop, topk_rank_static_k_unknown)
{
    const int64_t axis = 1;
    const auto data_shape = Shape{1, 10, 100};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    {
        const auto k = make_shared<op::Parameter>(element::i32, PartialShape({}));
        const auto topk = make_shared<TypeParam>(data, k, axis, "max", "value");

        const PartialShape fully_dynamic_axis_shape{1, Dimension(0, 10), 100};
        EXPECT_EQ(topk->get_output_partial_shape(0), fully_dynamic_axis_shape);
    }
    {
        const auto k = make_shared<op::v0::Constant>(element::i64, Shape{}, 5);
        const auto convert_k = make_shared<op::v0::Convert>(k, element::i32);
        const auto topk = make_shared<TypeParam>(data, convert_k, axis, "max", "value");

        const PartialShape ranged_dynamic_axis_shape{1, Dimension{5}, 100};
        EXPECT_EQ(topk->get_output_partial_shape(0), ranged_dynamic_axis_shape);
    }
}

REGISTER_TYPED_TEST_CASE_P(topk_type_prop,
                           topk_negative_axis_support,
                           topk_negative_axis_dynamic_rank,
                           topk_v1_partial_ouptut,
                           topk_rank_static_k_unknown,
                           topk_default_index_element_type);

typedef ::testing::Types<op::v1::TopK, op::v3::TopK> TopKTypes;
INSTANTIATE_TYPED_TEST_CASE_P(type_prop, topk_type_prop, TopKTypes, );
