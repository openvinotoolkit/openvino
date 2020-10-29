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

TEST(type_prop, tile)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    auto param1 = op::Constant::create(element::i64, Shape{3}, {3, 4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_shape(), (Shape{18, 32, 10}));
}

TEST(type_prop, tile_small_data_rank)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{8, 10});
    auto param1 = op::Constant::create(element::i64, Shape{3}, {3, 4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_shape(), (Shape{3, 32, 10}));
}

TEST(type_prop, tile_few_repeats)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{6, 8, 10});
    auto param1 = op::Constant::create(element::i64, Shape{2}, {4, 1});
    auto top = make_shared<op::v0::Tile>(param0, param1);
    ASSERT_EQ(top->get_element_type(), element::f32);
    ASSERT_EQ(top->get_shape(), (Shape{6, 32, 10}));
}
