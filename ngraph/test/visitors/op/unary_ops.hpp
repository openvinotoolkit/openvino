//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//  Licensed under the Apache License, Version 2.0 (the "License");
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

#pragma once

#include <vector>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

#include "ngraph/op/util/attr_types.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
// Unamed namespace

template <typename T>
class UnaryOperatorVisitor : public testing::Test
{
};

class UnaryOperatorTypeName
{
public:
    template <typename T>
    static std::string GetName(int)
    {
        const ngraph::Node::type_info_t typeinfo = T::get_type_info_static();
        return typeinfo.name;
    }
};

TYPED_TEST_CASE_P(UnaryOperatorVisitor);

TYPED_TEST_P(UnaryOperatorVisitor, No_Attribute_FP16_2D)
{
    NodeBuilder::get_ops().register_factory<TypeParam>();
    const auto A = std::make_shared<op::Parameter>(element::f16, PartialShape{5, 2});

    const auto op_func = std::make_shared<TypeParam>(A);
    NodeBuilder builder(op_func);
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TYPED_TEST_P(UnaryOperatorVisitor, No_Attribute_FP32_4D)
{
    NodeBuilder::get_ops().register_factory<TypeParam>();
    const auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 2, 2});

    const auto op_func = std::make_shared<TypeParam>(A);
    NodeBuilder builder(op_func);
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

REGISTER_TYPED_TEST_CASE_P(UnaryOperatorVisitor, No_Attribute_FP16_2D, No_Attribute_FP32_4D);
