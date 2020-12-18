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

#include "ngraph/op/softmax.hpp"

#include <algorithm>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg,
                         const HostTensorPtr& out,
                         const Shape& shape,
                         const AxisSet& axes)
    {
        runtime::reference::softmax(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape, axes);
        return true;
    }

    bool evaluate_softmax(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        auto shape = out->get_shape();
        bool rc = true;

        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_softmax, bf16, arg, out, shape, axes);
            NGRAPH_TYPE_CASE(evaluate_softmax, f16, arg, out, shape, axes);
            NGRAPH_TYPE_CASE(evaluate_softmax, f32, arg, out, shape, axes);
            NGRAPH_TYPE_CASE(evaluate_softmax, f64, arg, out, shape, axes);
        default: rc = false; break;
        }
        return rc;
    }
}

// *** SOFTMAX OP SET V1 ***
constexpr NodeTypeInfo op::v1::Softmax::type_info;

op::v1::Softmax::Softmax(const Output<Node>& arg, const size_t axis)
    : Op({arg})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Softmax::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v1::Softmax::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < input_shape.rank().get_length(),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::v1::Softmax::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Softmax>(new_args.at(0), m_axis);
}

bool op::v1::Softmax::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Softmax_evaluate, outputs[0]->set_unary(inputs[0]);
                    return evaluate_softmax(inputs[0], outputs[0], AxisSet{m_axis}));
    return false;
}
