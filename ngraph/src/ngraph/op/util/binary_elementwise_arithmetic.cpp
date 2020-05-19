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

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const AutoBroadcastSpec& autob)
    : m_autob(autob)
{
}

op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const Output<Node>& arg0,
                                                                   const Output<Node>& arg1,
                                                                   const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}

void op::util::BinaryElementwiseArithmetic::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic(m_autob);
}

bool op::util::BinaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}
