//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
    ASSERT_EQ(topk->get_axis(), data_shape.at(1));
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
        EXPECT_EQ(topk->get_output_partial_shape(0), PartialShape({2, -1}));
    }
    {
        auto k = make_shared<op::Constant>(element::i32, Shape{}, 3);
        auto topk = make_shared<TypeParam>(data, k, 1, "max", "value");
        EXPECT_EQ(topk->get_output_partial_shape(0), PartialShape({2, 3}));
    }
}

REGISTER_TYPED_TEST_CASE_P(topk_type_prop,
                           topk_negative_axis_support,
                           topk_negative_axis_dynamic_rank,
                           topk_v1_partial_ouptut);

typedef ::testing::Types<op::v1::TopK, op::v3::TopK> TopKTypes;
INSTANTIATE_TYPED_TEST_CASE_P(type_prop, topk_type_prop, TopKTypes, );
