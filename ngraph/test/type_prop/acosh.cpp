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

#include "unary_base.cpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

TEST(type_prop, acosh_basic_shape_inference)
{
    
    BasicShapeInference<op::Acosh>(element::f32, Shape{2, 2});
}

TEST(type_prop, acosh_incompatible_input_type)
{
    IncompatibleInputType<op::Acosh>(element::boolean, Shape{2, 2});
}

TEST(type_prop, acosh_dynamic_rank_input_shape)
{
    DynamicRankInputShape<op::Acosh>(element::f32);
}