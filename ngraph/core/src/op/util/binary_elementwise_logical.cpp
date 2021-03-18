//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/util/binary_elementwise_logical.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::BinaryElementwiseLogical, "BinaryElementwiseLogical", 0);

op::util::BinaryElementwiseLogical::BinaryElementwiseLogical() {}

op::util::BinaryElementwiseLogical::BinaryElementwiseLogical(const Output<Node>& arg0,
                                                             const Output<Node>& arg1,
                                                             const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}

void op::util::BinaryElementwiseLogical::validate_and_infer_elementwise_logical(
    const op::AutoBroadcastSpec& autob)
{
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this, autob);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(
        this,
        args_et.is_dynamic() || args_et == element::boolean,
        "Operands for logical operators must have boolean element type but have element type ",
        args_et,
        ".");

    set_output_type(0, element::boolean, args_pshape);
}

void op::util::BinaryElementwiseLogical::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_util_BinaryElementwiseLogical_validate_and_infer_types);
    validate_and_infer_elementwise_logical(m_autob);
}

bool op::util::BinaryElementwiseLogical::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_util_BinaryElementwiseLogical_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}
