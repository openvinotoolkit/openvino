// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/strided_slice.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/strided_slice.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

#include <algorithm>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::StridedSlice, "StridedSlice", 1);

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const Output<Node>& strides,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : Op({data, begin, end, strides})
    , m_begin_mask{begin_mask}
    , m_end_mask{end_mask}
    , m_new_axis_mask{new_axis_mask}
    , m_shrink_axis_mask{shrink_axis_mask}
    , m_ellipsis_mask{ellipsis_mask}
{
    constructor_validate_and_infer_types();
}

namespace
{
    shared_ptr<Node> calculate_default_strides(const Output<Node>& begin, const Output<Node>& end)
    {
        const auto begin_pshape = begin.get_partial_shape();
        const auto end_pshape = end.get_partial_shape();

        size_t strides_length = 0;
        if (begin_pshape.rank().is_static() && begin_pshape.rank().get_length() == 1 &&
            begin_pshape[0].is_static())
        {
            strides_length = begin_pshape[0].get_length();
        }
        else if (end_pshape.rank().is_static() && end_pshape.rank().get_length() == 1 &&
                 end_pshape[0].is_static())
        {
            strides_length = end_pshape[0].get_length();
        }
        else // dynamic case
        {
            NGRAPH_CHECK(begin_pshape.rank().is_static() && begin_pshape.rank().get_length() == 1,
                         "Begin input must be 1D");
            return std::make_shared<op::v1::Broadcast>(op::Constant::create(element::i64, {}, {1}),
                                                       std::make_shared<op::ShapeOf>(begin));
        }

        return op::Constant::create(
            element::i64, Shape{strides_length}, vector<int64_t>(strides_length, 1));
    }
} // namespace

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : StridedSlice(data,
                   begin,
                   end,
                   calculate_default_strides(begin, end),
                   begin_mask,
                   end_mask,
                   new_axis_mask,
                   shrink_axis_mask,
                   ellipsis_mask)
{
}

bool op::v1::StridedSlice::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_StridedSlice_visit_attributes);
    visitor.on_attribute("begin_mask", m_begin_mask);
    visitor.on_attribute("end_mask", m_end_mask);
    visitor.on_attribute("new_axis_mask", m_new_axis_mask);
    visitor.on_attribute("shrink_axis_mask", m_shrink_axis_mask);
    visitor.on_attribute("ellipsis_mask", m_ellipsis_mask);
    return true;
}

void op::v1::StridedSlice::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_StridedSlice_validate_and_infer_types);
    const auto& begin_mask_et = get_input_element_type(1);
    const auto& end_mask_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          begin_mask_et.is_integral_number(),
                          "Begin mask must be an integral number, but is: ",
                          begin_mask_et);
    NODE_VALIDATION_CHECK(this,
                          end_mask_et.is_integral_number(),
                          "End mask must be an integral number, but is: ",
                          end_mask_et);

    auto are_mask_elem_in_range = [](size_t e) { return e == 0 || e == 1; };
    NODE_VALIDATION_CHECK(
        this,
        std::all_of(m_begin_mask.begin(), m_begin_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_end_mask.begin(), m_end_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_new_axis_mask.begin(), m_new_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(
                m_shrink_axis_mask.begin(), m_shrink_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_ellipsis_mask.begin(), m_ellipsis_mask.end(), are_mask_elem_in_range),
        "All masks of StridedSlice must have be 0 or 1");

    const vector<size_t> attr_sizes = {m_begin_mask.size(),
                                       m_end_mask.size(),
                                       m_new_axis_mask.size(),
                                       m_shrink_axis_mask.size(),
                                       m_ellipsis_mask.size()};
    const auto are_attr_sizes_eq =
        std::all_of(attr_sizes.begin(), attr_sizes.end(), [&attr_sizes](size_t s) {
            return (s == 0) || (attr_sizes[0] == s);
        });
    NODE_VALIDATION_CHECK(
        this, are_attr_sizes_eq, "All masks of StridedSlice must have the same size");

    const auto& data_pshape = get_input_partial_shape(0);
    const auto& data_rank = data_pshape.rank();
    const auto& begin_shape = get_input_partial_shape(1);
    if (begin_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              begin_shape.rank().get_length() == 1,
                              "Begin input must be 1D (begin rank: ",
                              begin_shape.rank(),
                              ").");
    }
    const auto& end_shape = get_input_partial_shape(2);
    if (end_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              end_shape.rank().get_length() == 1,
                              "End input must be 1D (end rank: ",
                              end_shape.rank(),
                              ").");
    }

    // Fill up strides input with default strides if not set by this point.
    if (get_input_size() < 4)
    {
        set_argument(3,
                     calculate_default_strides(get_input_node_ptr(1)->output(0),
                                               get_input_node_ptr(2)->output(0)));
    }

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    if (data_rank.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    auto begin_const = get_constant_from_source(input_value(1));
    auto end_const = get_constant_from_source(input_value(2));
    auto strides = get_constant_from_source(input_value(3));

    const auto& new_axis_axis_set = convert_mask_to_axis_set(get_new_axis_mask());
    const auto& shrink_axis_axis_set = convert_mask_to_axis_set(get_shrink_axis_mask());
    const auto& ellipsis_axis_set = convert_mask_to_axis_set(get_ellipsis_mask());

    if (begin_const && end_const && strides)
    {
        const auto& input_pshape = get_input_partial_shape(0);

        const auto& begin_vector = begin_const->cast_vector<int64_t>();
        const auto& end_vector = end_const->cast_vector<int64_t>();
        const auto& strides_vector = strides->cast_vector<int64_t>();

        const auto& begin_axis_set = convert_mask_to_axis_set(get_begin_mask());
        const auto& end_axis_set = convert_mask_to_axis_set(get_end_mask());

        PartialShape output_pshape;
        if (input_pshape.is_static())
            output_pshape = infer_slice_shape(this,
                                              input_pshape,
                                              begin_vector,
                                              end_vector,
                                              strides_vector,
                                              begin_axis_set,
                                              end_axis_set,
                                              new_axis_axis_set,
                                              shrink_axis_axis_set,
                                              ellipsis_axis_set);
        else
        {
            // output shape dimensions of ss operation depend monotonically on the input shape
            // dimensions: the larger the input dimensions the larger the output dimensions in order
            // to use existing shape infer function of strided slice we split each dimension into
            // lower and upper ones and put them into lower input shape and upper input shape
            // correspondingly Example: data_shape = {{5, 9}, {0, 3}, -1, 5}, lower_shape = {5, 0,
            // 0, 5}, upper_shape = {9, 3, max_int64, 5} With that we calculate two output shapes as
            // if there were two strided slices but with different input shapes and combine them
            // into one resulting output shape

            auto lower_in_shape = PartialShape::dynamic(input_pshape.rank()),
                 upper_in_shape = PartialShape::dynamic(input_pshape.rank());
            for (auto i = 0; i < data_rank.get_length(); ++i)
            {
                lower_in_shape[i] = input_pshape[i].get_min_length();
                upper_in_shape[i] = input_pshape[i].get_max_length();
            }

            auto lower_out_shape = infer_slice_shape(this,
                                                     lower_in_shape,
                                                     begin_vector,
                                                     end_vector,
                                                     strides_vector,
                                                     begin_axis_set,
                                                     end_axis_set,
                                                     new_axis_axis_set,
                                                     shrink_axis_axis_set,
                                                     ellipsis_axis_set);

            auto upper_out_shape = infer_slice_shape(this,
                                                     upper_in_shape,
                                                     begin_vector,
                                                     end_vector,
                                                     strides_vector,
                                                     begin_axis_set,
                                                     end_axis_set,
                                                     new_axis_axis_set,
                                                     shrink_axis_axis_set,
                                                     ellipsis_axis_set);
            NODE_VALIDATION_CHECK(this,
                                  lower_out_shape.rank().is_static() &&
                                      upper_out_shape.rank().is_static() &&
                                      lower_out_shape.rank() == upper_out_shape.rank());

            output_pshape = PartialShape::dynamic(lower_in_shape.rank());
            for (auto i = 0; i < lower_out_shape.rank().get_length(); ++i)
            {
                auto minimum =
                    lower_out_shape[i].is_static() ? lower_out_shape[i].get_length() : -1;
                auto maximum =
                    upper_out_shape[i].is_static() ? upper_out_shape[i].get_length() : -1;
                output_pshape[i] = {minimum, maximum};
            }
        }
        set_output_type(0, get_input_element_type(0), output_pshape);
    }
    else
    {
        if (new_axis_axis_set.empty() && shrink_axis_axis_set.empty() && ellipsis_axis_set.empty())
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic(data_rank));
        else
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

AxisSet op::v1::StridedSlice::convert_mask_to_axis_set(const std::vector<int64_t>& mask) const
{
    AxisSet axis_set{};
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i)
    {
        if (mask[i] == 1)
        {
            axis_set.emplace(i);
        }
    }
    return axis_set;
}

shared_ptr<Node> op::v1::StridedSlice::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_StridedSlice_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::StridedSlice>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         m_begin_mask,
                                         m_end_mask,
                                         m_new_axis_mask,
                                         m_shrink_axis_mask,
                                         m_ellipsis_mask);
}

namespace strided_slice
{
    inline bool evaluate(const HostTensorPtr& in, const SlicePlan& sp, const HostTensorPtr& out)

    {
        auto in_shape = in->get_shape();
        out->set_shape(sp.reshape_out_shape);
        runtime::reference::strided_slice(in->get_data_ptr<char>(),
                                          out->get_data_ptr<char>(),
                                          in_shape,
                                          sp,
                                          in->get_element_type().size());
        return true;
    }

    bool evaluate_strided_slice(const HostTensorPtr& in,
                                const HostTensorPtr& begin,
                                const HostTensorPtr& end,
                                const HostTensorPtr& stride,
                                const AxisSet& begin_mask,
                                const AxisSet& end_mask,
                                const AxisSet& new_axis_mask,
                                const AxisSet& shrink_axis_mask,
                                const AxisSet& ellipsis_mask,
                                const HostTensorPtr& out)
    {
        std::vector<int64_t> begin_const = host_tensor_2_vector<int64_t>(begin);
        std::vector<int64_t> end_const = host_tensor_2_vector<int64_t>(end);
        std::vector<int64_t> stride_const = host_tensor_2_vector<int64_t>(stride);
        SlicePlan slice_plan = make_slice_plan(in->get_shape(),
                                               begin_const,
                                               end_const,
                                               stride_const,
                                               begin_mask,
                                               end_mask,
                                               new_axis_mask,
                                               shrink_axis_mask,
                                               ellipsis_mask);
        return evaluate(in, slice_plan, out);
    }
} // namespace strided_slice

bool op::v1::StridedSlice::evaluate(const HostTensorVector& output_values,
                                    const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v1_StridedSlice_evaluate);
    // FIXME: 4th input is optional, but it is required by the following code
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 4));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    return strided_slice::evaluate_strided_slice(input_values[0],
                                                 input_values[1],
                                                 input_values[2],
                                                 input_values[3],
                                                 convert_mask_to_axis_set(get_begin_mask()),
                                                 convert_mask_to_axis_set(get_end_mask()),
                                                 convert_mask_to_axis_set(get_new_axis_mask()),
                                                 convert_mask_to_axis_set(get_shrink_axis_mask()),
                                                 convert_mask_to_axis_set(get_ellipsis_mask()),
                                                 output_values[0]);
}

bool op::v1::StridedSlice::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_StridedSlice_has_evaluate);
    return get_input_size() == 4;
}

bool op::v1::StridedSlice::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        !input_value(2).get_tensor().has_and_set_bound() ||
        !input_value(3).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v1::StridedSlice::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        !input_value(2).get_tensor().has_and_set_bound() ||
        !input_value(3).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}
