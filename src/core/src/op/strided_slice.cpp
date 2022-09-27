// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/strided_slice.hpp"

#include <algorithm>

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
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "strided_slice_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::StridedSlice);

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const Output<Node>& strides,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : Op({data, begin, end, strides}),
      m_begin_mask{begin_mask},
      m_end_mask{end_mask},
      m_new_axis_mask{new_axis_mask},
      m_shrink_axis_mask{shrink_axis_mask},
      m_ellipsis_mask{ellipsis_mask} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    ov::mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}

namespace {
shared_ptr<Node> calculate_default_strides(const Output<Node>& begin, const Output<Node>& end) {
    const auto begin_pshape = begin.get_partial_shape();
    const auto end_pshape = end.get_partial_shape();

    size_t strides_length = 0;
    if (begin_pshape.rank().is_static() && begin_pshape.rank().get_length() == 1 && begin_pshape[0].is_static()) {
        strides_length = begin_pshape[0].get_length();
    } else if (end_pshape.rank().is_static() && end_pshape.rank().get_length() == 1 && end_pshape[0].is_static()) {
        strides_length = end_pshape[0].get_length();
    } else  // dynamic case
    {
        NGRAPH_CHECK(begin_pshape.rank().is_static() && begin_pshape.rank().get_length() == 1,
                     "Begin input must be 1D");
        return std::make_shared<op::v1::Broadcast>(op::Constant::create(element::i64, {}, {1}),
                                                   std::make_shared<op::ShapeOf>(begin));
    }

    return op::Constant::create(element::i64, ov::Shape{strides_length}, vector<int64_t>(strides_length, 1));
}
}  // namespace

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
                   ellipsis_mask) {}

bool op::v1::StridedSlice::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_StridedSlice_visit_attributes);
    visitor.on_attribute("begin_mask", m_begin_mask);
    visitor.on_attribute("end_mask", m_end_mask);
    visitor.on_attribute("new_axis_mask", m_new_axis_mask);
    visitor.on_attribute("shrink_axis_mask", m_shrink_axis_mask);
    visitor.on_attribute("ellipsis_mask", m_ellipsis_mask);
    return true;
}

void op::v1::StridedSlice::validate_and_infer_types() {
    OV_OP_SCOPE(v1_StridedSlice_validate_and_infer_types);
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

    auto are_mask_elem_in_range = [](size_t e) {
        return e == 0 || e == 1;
    };
    NODE_VALIDATION_CHECK(
        this,
        std::all_of(m_begin_mask.begin(), m_begin_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_end_mask.begin(), m_end_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_new_axis_mask.begin(), m_new_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_shrink_axis_mask.begin(), m_shrink_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_ellipsis_mask.begin(), m_ellipsis_mask.end(), are_mask_elem_in_range),
        "All masks of StridedSlice must have be 0 or 1");

    const vector<size_t> attr_sizes = {m_begin_mask.size(),
                                       m_end_mask.size(),
                                       m_new_axis_mask.size(),
                                       m_shrink_axis_mask.size(),
                                       m_ellipsis_mask.size()};
    const auto are_attr_sizes_eq = std::all_of(attr_sizes.begin(), attr_sizes.end(), [&attr_sizes](size_t s) {
        return (s == 0) || (attr_sizes[0] == s);
    });
    NODE_VALIDATION_CHECK(this, are_attr_sizes_eq, "All masks of StridedSlice must have the same size");

    // Fill up strides input with default strides if not set by this point.
    if (get_input_size() < 4) {
        set_argument(3, calculate_default_strides(get_input_node_ptr(1)->output(0), get_input_node_ptr(2)->output(0)));
    }

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape::dynamic()};
    for (size_t input_idx = 0; input_idx < get_input_size(); ++input_idx) {
        input_shapes.push_back(get_input_partial_shape(input_idx));
    }

    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

AxisSet op::v1::StridedSlice::convert_mask_to_axis_set(const std::vector<int64_t>& mask) const {
    AxisSet axis_set{};
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
        if (mask[i] == 1) {
            axis_set.emplace(i);
        }
    }
    return axis_set;
}

shared_ptr<Node> op::v1::StridedSlice::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_StridedSlice_clone_with_new_inputs);
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

namespace strided_slice {
namespace {
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
                            const HostTensorPtr& out) {
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
}  // namespace
}  // namespace strided_slice

bool op::v1::StridedSlice::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v1_StridedSlice_evaluate);
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

bool op::v1::StridedSlice::has_evaluate() const {
    OV_OP_SCOPE(v1_StridedSlice_has_evaluate);
    return get_input_size() == 4;
}

namespace {
bool strided_slice_input_check(const ov::Node* node) {
    if (!node->get_input_tensor(1).has_and_set_bound() || !node->get_input_tensor(2).has_and_set_bound() ||
        !node->get_input_tensor(3).has_and_set_bound())
        return false;
    return true;
}
}  // namespace

bool op::v1::StridedSlice::evaluate_lower(const HostTensorVector& output_values) const {
    if (!strided_slice_input_check(this))
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v1::StridedSlice::evaluate_upper(const HostTensorVector& output_values) const {
    if (!strided_slice_input_check(this))
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v1::StridedSlice::evaluate_label(TensorLabelVector& output_labels) const {
    if (!strided_slice_input_check(this))
        return false;
    return default_label_evaluator(this, output_labels);
}
