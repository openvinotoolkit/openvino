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

#include "ngraph/op/one_hot.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::OneHot::type_info;

op::v1::OneHot::OneHot(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis)
    : Op({indices, depth, on_value, off_value})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

void op::v1::OneHot::validate_and_infer_types()
{
    const auto& indices_et = get_input_element_type(0);
    const auto& depth_et = get_input_element_type(1);
    const auto& on_value_et = get_input_element_type(2);
    const auto& off_value_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_dynamic() || indices_et.is_integral(),
                          "Indices must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          depth_et.is_dynamic() || depth_et.is_integral(),
                          "Depth must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          on_value_et.compatible(off_value_et),
                          "on_value element type must be compatible with off_value element type.");

    const auto& indices_shape = get_input_partial_shape(0);
    const auto& depth_shape = get_input_partial_shape(1);
    const auto& on_value_shape = get_input_partial_shape(2);
    const auto& off_value_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          depth_shape.is_dynamic() || is_scalar(depth_shape.to_shape()),
                          "depth input must be scalar.");

    NODE_VALIDATION_CHECK(this,
                          on_value_shape.is_dynamic() || is_scalar(on_value_shape.to_shape()),
                          "on_value input must be scalar.");

    NODE_VALIDATION_CHECK(this,
                          off_value_shape.is_dynamic() || is_scalar(off_value_shape.to_shape()),
                          "off_value input must be scalar.");

    const auto& depth = input_value(1).get_node_shared_ptr();
    PartialShape result_shape{PartialShape::dynamic()};

    if (indices_shape.is_static() && indices_shape.rank().is_static() && op::is_constant(depth))
    {
        const auto indices_rank = indices_shape.rank().get_length();

        std::vector<Dimension> out_dims(indices_rank);
        for (auto i = 0; i < indices_rank; i++)
        {
            out_dims[i] = indices_shape[i];
        }
        m_axis =
            ngraph::normalize_axis(this, m_axis, indices_rank + 1, -indices_rank - 1, indices_rank);

        auto depth_element_type = depth->get_output_element_type(0);
        NODE_VALIDATION_CHECK(this,
                              depth_element_type.is_integral(),
                              "'depth' input element type must be an integer (got ",
                              depth_element_type,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              is_scalar(depth->get_shape()),
                              "A scalar input should be provided as 'depth' to OneHot",
                              " (got ",
                              depth->get_shape(),
                              " elements).");

        const auto depth_constant = as_type_ptr<op::Constant>(depth);
        int64_t depth_val = depth_constant->cast_vector<int64_t>()[0];

        NODE_VALIDATION_CHECK(this,
                              depth_val > 0,
                              "The value of 'depth' must be a positive number.",
                              " (got ",
                              depth_val,
                              ").");

        out_dims.insert(out_dims.begin() + m_axis, Dimension(depth_val));
        result_shape = out_dims;
    }

    set_output_type(0, on_value_et, result_shape);
}

bool ngraph::op::v1::OneHot::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v1::OneHot::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::OneHot>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_axis);
}

namespace detail
{
    template <typename ind_t, typename out_t>
    void evaluate(const HostTensorVector& output_values,
                  const HostTensorVector& input_values,
                  const int64_t axis)
    {
        const auto& indices = input_values[0];
        const auto& depth = input_values[1];
        const auto& on_value = input_values[2];
        const auto& off_value = input_values[3];

        const auto& out = output_values[0];

        runtime::reference::one_hot<ind_t, out_t>(indices->get_data_ptr<ind_t>(),
                                                  out->get_data_ptr<out_t>(),
                                                  indices->get_shape(),
                                                  out->get_shape(),
                                                  axis,
                                                  on_value->get_data_ptr<out_t>()[0],
                                                  off_value->get_data_ptr<out_t>()[0]);
    }

    template <typename out_t>
    bool dispatch_by_output_type(const HostTensorVector& output_values,
                                 const HostTensorVector& input_values,
                                 const int64_t axis)
    {
        const auto& indices = input_values[0];

        switch (indices->get_element_type())
        {
        case element::Type_t::i32:
            evaluate<int32_t, out_t>(output_values, input_values, axis);
            break;
        case element::Type_t::i64:
            evaluate<int64_t, out_t>(output_values, input_values, axis);
            break;
        default: return false; break;
        }

        return true;
    }

    bool evaluate_onehot(const HostTensorVector& output_values,
                         const HostTensorVector& input_values,
                         const int64_t axis)
    {
        const auto& on_value = input_values[2];

        switch (on_value->get_element_type())
        {
        case element::Type_t::boolean:
            return dispatch_by_output_type<char>(output_values, input_values, axis);
            break;
        case element::Type_t::f32:
            return dispatch_by_output_type<float>(output_values, input_values, axis);
            break;
        case element::Type_t::i32:
            return dispatch_by_output_type<int32_t>(output_values, input_values, axis);
            break;
        case element::Type_t::i64:
            return dispatch_by_output_type<int64_t>(output_values, input_values, axis);
            break;
        default: return false;
        }
    }
} // namespace detail

bool op::v1::OneHot::evaluate(const HostTensorVector& output_values,
                              const HostTensorVector& input_values) const
{
    return detail::evaluate_onehot(output_values, input_values, get_axis());
}
