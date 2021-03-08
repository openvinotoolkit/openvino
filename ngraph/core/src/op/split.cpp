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
#include "ngraph/runtime/reference/split.hpp"
#include <numeric>
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/split.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::Split, "Split", 1);

op::v1::Split::Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits)
    : Op({data, axis})
    , m_num_splits{num_splits}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Split::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Split_visit_attributes);
    visitor.on_attribute("num_splits", m_num_splits);
    return true;
}

void op::v1::Split::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Split_validate_and_infer_types);
    const auto data_ps = input_value(0).get_partial_shape();
    const auto axis_ps = input_value(1).get_partial_shape();
    const auto axis_et = input_value(1).get_element_type();

    if (axis_ps.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              axis_ps.rank().get_length() == 0,
                              "The 'axis' input is expected to be a scalar. Got: ",
                              axis_ps);
    }

    NODE_VALIDATION_CHECK(
        this, axis_et.is_integral(), "The 'axis' input only accepts integral types");

    PartialShape each_output_shape{data_ps};
    const auto axis_input = get_constant_from_source(input_value(1));
    if (axis_input && data_ps.rank().is_static())
    {
        auto axis = axis_input->cast_vector<int64_t>()[0];

        const auto data_rank = get_input_partial_shape(0).rank();
        axis = ngraph::normalize_axis(this, axis, data_rank);

        if (data_ps[axis].is_static())
        {
            const auto dimension_at_axis = data_ps[axis].get_length();

            NODE_VALIDATION_CHECK(this,
                                  dimension_at_axis % m_num_splits == 0,
                                  "The input tensor's dimension pointed by the 'axis' parameter: ",
                                  dimension_at_axis,
                                  " has to be a multiple of the 'num_splits' attribute value: ",
                                  m_num_splits);

            each_output_shape[axis] = dimension_at_axis / m_num_splits;
        }
        else
        {
            each_output_shape[axis] = Dimension::dynamic();
        }
    }
    else
    {
        each_output_shape = PartialShape::dynamic(data_ps.rank());
    }

    for (size_t i = 0; i < m_num_splits; ++i)
    {
        set_output_type(i, get_input_element_type(0), each_output_shape);
    }

    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v1::Split::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Split_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Split>(new_args.at(0), new_args.at(1), m_num_splits);
}

namespace split
{
    inline bool evaluate(const HostTensorPtr& data_tensor,
                         const HostTensorVector& outputs,
                         const int64_t axis,
                         const int64_t num_splits)
    {
        Shape output_shape = data_tensor->get_shape();
        std::vector<char*> outputs_data(num_splits);
        output_shape.at(axis) /= num_splits;
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            outputs[i]->set_shape(output_shape);
            outputs_data[i] = outputs[i]->get_data_ptr<char>();
        }
        ngraph::runtime::reference::split(data_tensor->get_data_ptr<char>(),
                                          data_tensor->get_shape(),
                                          data_tensor->get_element_type().size(),
                                          axis,
                                          num_splits,
                                          outputs_data.data());
        return true;
    }

    bool evaluate_split(const HostTensorPtr& data_tensor,
                        const HostTensorPtr& axis_tensor,
                        const HostTensorVector& outputs,
                        const int64_t num_splits,
                        const Node* split_node)
    {
        NGRAPH_CHECK(axis_tensor->get_element_type().is_integral_number(),
                     "axis element type is not integral data type");

        int64_t axis = host_tensor_2_vector<int64_t>(axis_tensor)[0];

        axis = ngraph::normalize_axis(split_node, axis, data_tensor->get_partial_shape().rank());
        evaluate(data_tensor, outputs, axis, num_splits);
        return true;
    }
}

bool op::v1::Split::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Split_evaluate);
    const auto& data = inputs[0];
    const auto& axis = inputs[1];
    return split::evaluate_split(data, axis, outputs, m_num_splits, this);
}
