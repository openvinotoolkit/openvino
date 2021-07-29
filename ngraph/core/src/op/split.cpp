// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    const PartialShape& data_ps = get_input_partial_shape(0);
    const PartialShape& axis_ps = get_input_partial_shape(1);
    const element::Type& axis_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(
        this, axis_ps.rank().compatible(0), "'axis' input must be a scalar. Got: ", axis_ps);

    NODE_VALIDATION_CHECK(this,
                          axis_et.is_integral_number(),
                          "Element type of 'axis' input must be integer. Got: ",
                          axis_et);

    NODE_VALIDATION_CHECK(this,
                          m_num_splits > 0,
                          "Attribute 'num_splits' must be greater than zero. Got: ",
                          m_num_splits);

    PartialShape each_output_shape{data_ps};
    const Rank data_rank = data_ps.rank();
    const auto axis_input = get_constant_from_source(input_value(1));
    if (axis_input && data_rank.is_static())
    {
        auto axis = axis_input->cast_vector<int64_t>()[0];
        axis = ngraph::normalize_axis(this, axis, data_rank);

        if (data_ps[axis].is_static())
        {
            const auto dimension_at_axis = data_ps[axis].get_length();

            NODE_VALIDATION_CHECK(this,
                                  dimension_at_axis % m_num_splits == 0,
                                  "Dimension of data input shape along 'axis': ",
                                  dimension_at_axis,
                                  " must be evenly divisible by 'num_splits' attribute value: ",
                                  m_num_splits);

            each_output_shape[axis] = dimension_at_axis / m_num_splits;
        }
        else
        {
            const auto dim_interval_at_axis = data_ps[axis].get_interval();
            NODE_VALIDATION_CHECK(
                this,
                dim_interval_at_axis.get_max_val() >= static_cast<int64_t>(m_num_splits),
                "The interval maximum of the dimension for data input shape along 'axis' must be "
                "greater or equal to 'num_splits' attribute. Got: ",
                dim_interval_at_axis,
                " and ",
                m_num_splits);

            auto dim_interval_at_axis_min =
                static_cast<int64_t>(dim_interval_at_axis.get_min_val() * (1.0f / m_num_splits));
            auto dim_interval_at_axis_max = dim_interval_at_axis.get_max_val();
            if (dim_interval_at_axis.has_upper_bound())
            {
                dim_interval_at_axis_max =
                    static_cast<int64_t>(dim_interval_at_axis_max * (1.0f / m_num_splits));
            }
            each_output_shape[axis] = Dimension(dim_interval_at_axis_min, dim_interval_at_axis_max);
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
} // namespace split

bool op::v1::Split::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Split_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, m_num_splits) &&
                 validate_host_tensor_vector(inputs, 2));
    const auto& data = inputs[0];
    const auto& axis = inputs[1];
    return split::evaluate_split(data, axis, outputs, m_num_splits, this);
}

bool op::v1::Split::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Split_has_evaluate);
    return get_input_element_type(1).is_integral_number();
}
