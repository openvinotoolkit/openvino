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

#include "ngraph/op/one_hot.hpp"
#include "itt.hpp"
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
    NGRAPH_OP_SCOPE(v1_OneHot_validate_and_infer_types);
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

    PartialShape result_shape{PartialShape::dynamic()};
    const auto& depth = input_value(1).get_node_shared_ptr();
    const auto& depth_constant = get_constant_from_source(input_value(1));
    if (indices_shape.rank().is_static() && depth_constant)
    {
        std::vector<Dimension> out_dims{indices_shape};
        const auto indices_rank = indices_shape.rank().get_length();
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
    NGRAPH_OP_SCOPE(v1_OneHot_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v1::OneHot::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_OneHot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::OneHot>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_axis);
}

namespace detail
{
    template <typename ind_t, typename out_t>
    bool evaluate(const HostTensorVector& output_values,
                  const HostTensorVector& input_values,
                  const int64_t axis)
    {
        const auto& indices = input_values[0];
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
        return true;
    }

#define TYPE_OUT_CASE(a, ...)                                                                      \
    case element::Type_t::a:                                                                       \
    {                                                                                              \
        NGRAPH_OP_SCOPE(OV_CC_CAT3(evaluate_one_hot_out, _, a));                                   \
        using IT = typename element_type_traits<element::Type_t::a>::value_type;                   \
        using OT = typename element_type_traits<out_t>::value_type;                                \
        rc = evaluate<IT, OT>(__VA_ARGS__);                                                        \
    }                                                                                              \
    break

    template <element::Type_t out_t>
    bool evaluate(const HostTensorVector& output_values,
                  const HostTensorVector& input_values,
                  const int64_t axis)
    {
        const auto& indices = input_values[0];

        bool rc = true;
        switch (indices->get_element_type())
        {
            TYPE_OUT_CASE(i32, output_values, input_values, axis);
            TYPE_OUT_CASE(i64, output_values, input_values, axis);
        default: rc = false; break;
        }

        return rc;
    }

    bool evaluate_onehot(const HostTensorVector& output_values,
                         const HostTensorVector& input_values,
                         const int64_t axis)
    {
        const auto& on_value = input_values[2];

        bool rc = false;
        switch (on_value->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_onehot, boolean, output_values, input_values, axis);
            NGRAPH_TYPE_CASE(evaluate_onehot, f32, output_values, input_values, axis);
            NGRAPH_TYPE_CASE(evaluate_onehot, i32, output_values, input_values, axis);
            NGRAPH_TYPE_CASE(evaluate_onehot, i64, output_values, input_values, axis);
        default: rc = false;
        }
        return rc;
    }
} // namespace detail

bool op::v1::OneHot::evaluate(const HostTensorVector& output_values,
                              const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v1_OneHot_evaluate);
    return detail::evaluate_onehot(output_values, input_values, get_axis());
}
