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

#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace std;
using namespace ngraph;

op::util::BinaryElementwiseComparison::BinaryElementwiseComparison(const AutoBroadcastSpec& autob)
    : m_autob(autob)
{
}

op::util::BinaryElementwiseComparison::BinaryElementwiseComparison(const Output<Node>& arg0,
                                                                   const Output<Node>& arg1,
                                                                   const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}

void op::util::BinaryElementwiseComparison::validate_and_infer_types()
{
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this, m_autob);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    set_output_type(0, element::boolean, args_pshape);
}

bool op::util::BinaryElementwiseComparison::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}
