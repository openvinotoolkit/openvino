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

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace ngraph;

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic()
    : Op()
{
}

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const Output<Node>& arg)
    : Op({arg})
{
}

void op::util::UnaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic()
{
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et != element::boolean,
                          "Arguments cannot have boolean element type (argument element type: ",
                          args_et,
                          ").");

    set_output_type(0, args_et, args_pshape);
}

void op::util::UnaryElementwiseArithmetic::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic();
}

bool op::util::UnaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}
