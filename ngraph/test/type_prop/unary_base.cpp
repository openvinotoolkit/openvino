//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

template <class Op>
void BasicShapeInference(ngraph::element::Type_t type, Shape shape)
{
    const auto param = make_shared<op::Parameter>(type, shape);
    const auto op = make_shared<Op>(param);
    ASSERT_EQ(op->get_shape(), (shape));
    ASSERT_EQ(op->get_element_type(), type);
};

template <class Op>
void IncompatibleInputType(ngraph::element::Type_t type, Shape shape)
{
    const auto param = make_shared<op::Parameter>(type, shape);
    ASSERT_THROW(make_shared<Op>(param), ngraph::NodeValidationFailure);
}

template <class Op>
void DynamicRankInputShape(ngraph::element::Type_t type)
{
    const auto param = make_shared<op::Parameter>(type, PartialShape::dynamic());
    const auto op = make_shared<Op>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
