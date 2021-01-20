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

#include <cmath>
#include "itt.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::Gelu::type_info;

op::Gelu::Gelu()
    : FusedOp()
{
}

op::Gelu::Gelu(const Output<Node>& data)
    : FusedOp({data})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Gelu::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Gelu_visit_attributes);
    return true;
}

// f(x) = 0.5 * x * (1.0 + erf( x / sqrt(2.0) )
OutputVector op::Gelu::decompose_op() const
{
    auto data = input_value(0);

    shared_ptr<ngraph::Node> half =
        builder::make_constant(data.get_element_type(), data.get_shape(), 0.5);

    shared_ptr<ngraph::Node> one =
        builder::make_constant(data.get_element_type(), data.get_shape(), 1.0);

    shared_ptr<ngraph::Node> sqrt_two =
        builder::make_constant(data.get_element_type(), data.get_shape(), std::sqrt(2.0));

    shared_ptr<ngraph::Node> add = std::make_shared<op::v1::Add>(
        one, make_shared<ngraph::op::Erf>(std::make_shared<op::v1::Divide>(data, sqrt_two)));
    shared_ptr<ngraph::Node> multiply = std::make_shared<op::v1::Multiply>(half, data);

    return {std::make_shared<op::v1::Multiply>(multiply, add)};
}

shared_ptr<Node> op::Gelu::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Gelu>(new_args.at(0));
}

void op::Gelu::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (input_pshape.is_dynamic())
    {
        set_output_type(0, input_element_type, input_pshape);
    }
}
