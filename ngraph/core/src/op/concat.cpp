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

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/concat.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::Concat, "Concat", 0);

op::Concat::Concat(const OutputVector& args, int64_t axis)
    : Op(args)
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

op::Concat::Concat(const NodeVector& args, int64_t axis)
    : Concat(as_output_vector(args), axis)
{
}

bool op::Concat::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::Concat::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this, get_input_size() >= 1, "At least one argument required.");

    PartialShape inputs_shape_scheme{PartialShape::dynamic()};
    element::Type inputs_et{element::Type_t::dynamic};
    Dimension concatenation_axis_output_dim{0};

    for (uint64_t i = 0; i < get_input_size(); i++)
    {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
        PartialShape this_input_shape = get_input_partial_shape(i);
        Dimension this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static())
        {
            if (get_concatenation_axis() < 0)
            {
                set_concatenation_axis(get_axis() < 0 ? get_axis() + this_input_rank.get_length()
                                                      : get_axis());
            }
            auto concat_axis = get_concatenation_axis();
            NODE_VALIDATION_CHECK(this,
                                  concat_axis < this_input_rank.get_length(),
                                  "Concatenation axis (",
                                  concat_axis,
                                  ") is out of bounds for ",
                                  "argument ",
                                  i,
                                  ", which has shape ",
                                  this_input_shape,
                                  ".");

            concatenation_axis_output_dim += this_input_shape[concat_axis];
            this_input_shape[concat_axis] = Dimension::dynamic();

            NODE_VALIDATION_CHECK(
                this,
                PartialShape::merge_into(inputs_shape_scheme, this_input_shape),
                "Argument shapes are inconsistent; they must have the same rank, and must have ",
                "equal dimension everywhere except on the concatenation axis (axis ",
                concat_axis,
                ").");
        }
        else
        {
            concatenation_axis_output_dim += Dimension::dynamic();
        }
    }
    PartialShape concatenated_shape = inputs_shape_scheme;

    if (concatenated_shape.rank().is_static())
    {
        concatenated_shape[get_concatenation_axis()] = concatenation_axis_output_dim;
        set_output_type(0, inputs_et, concatenated_shape);
    }
    else
    {
        set_output_type(0, inputs_et, PartialShape::dynamic(concatenation_axis_output_dim));
    }
}

shared_ptr<Node> op::Concat::clone_with_new_inputs(const OutputVector& new_args) const
{
    // TODO(amprocte): Should we check the new_args count here?
    return make_shared<Concat>(new_args, m_axis);
}

namespace
{
    bool evaluate_concat(const HostTensorVector& args,
                         const HostTensorPtr& out,
                         int64_t concatenation_axis)
    {
        std::vector<const char*> arg_bufs;
        std::vector<Shape> arg_shapes;
        Shape out_shape(args[0]->get_shape());
        out_shape[concatenation_axis] = 0;
        for (auto& input : args)
        {
            arg_bufs.push_back(input->get_data_ptr<char>());
            arg_shapes.push_back(input->get_shape());
            out_shape[concatenation_axis] += arg_shapes.back()[concatenation_axis];
        }
        out->set_shape(out_shape);
        runtime::reference::concat(arg_bufs,
                                   out->get_data_ptr<char>(),
                                   arg_shapes,
                                   out_shape,
                                   concatenation_axis,
                                   out->get_element_type().size());

        return true;
    }
}

bool op::Concat::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::Concat::evaluate");
    auto concat_axis = get_axis() < 0 ? get_axis() + inputs[0]->get_shape().size() : get_axis();
    return evaluate_concat(inputs, outputs[0], concat_axis);
}
