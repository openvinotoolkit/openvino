// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/slice.hpp"

#include <numeric>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::Slice, "Slice", 8);

op::v8::Slice::Slice(const Output<Node>& data,
                     const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step)
    : Op({data, start, stop, step}) {
    constructor_validate_and_infer_types();
}

op::v8::Slice::Slice(const Output<Node>& data,
                     const Output<Node>& start,
                     const Output<Node>& stop,
                     const Output<Node>& step,
                     const Output<Node>& axes)
    : Op({data, start, stop, step, axes}) {
    constructor_validate_and_infer_types();
}

namespace {

std::shared_ptr<ngraph::op::v0::Constant> get_default_const_axes(const Output<Node>& start) {
    const auto start_pshape = start.get_partial_shape();
    // Static case
    if (start_pshape.rank().is_static() && start_pshape.rank().get_length() == 1 && start_pshape[0].is_static()) {
        size_t axes_length = start_pshape[0].get_length();
        std::vector<int64_t> axes(axes_length);
        std::iota(axes.begin(), axes.end(), 0);
        return op::v0::Constant::create(element::i64, Shape{axes_length}, axes);
    } else
        return nullptr;  // Dynamic case, if start is parameter without const values, we can't calculate output dims
                         // anyway;

    // // Dynamic case // not needed, if start is dynamic we can't calculate output dims anyway
    // const auto axes_start_val = op::Constant::create(element::i64, {}, {0});
    // const auto axes_end_val = std::make_shared<op::v0::Squeeze>(std::make_shared<op::v3::ShapeOf>(start));
    // const auto axes_step_val = op::Constant::create(element::i64, {}, {1});
    // return std::make_shared<op::v4::Range>(axes_start_val, axes_end_val, axes_step_val, element::i64);
}
}  // namespace

bool op::v8::Slice::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_Slice_visit_attributes);
    return true;
}

void op::v8::Slice::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v8_Slice_validate_and_infer_types);

    const PartialShape& data_shape = get_input_partial_shape(0);
    PartialShape output_shape(data_shape);

    // If data_shape.rank() is dynamic we can't calulate output shape
    // even with const start/stop/step/axes we don't know how many axes should be copied
    // as "unspefified" in the final output shape, so the output shape rank is also dynamic.
    if (data_shape.rank().is_dynamic()) {
        set_output_type(0, get_input_element_type(0), output_shape);
    }

    const auto& start_rank = get_input_partial_shape(1).rank();
    const auto& stop_rank = get_input_partial_shape(2).rank();
    const auto& step_rank = get_input_partial_shape(3).rank();

    NODE_VALIDATION_CHECK(this, start_rank.compatible(1), "Start input must be a 1D tensor. Got: ", start_rank);
    NODE_VALIDATION_CHECK(this, stop_rank.compatible(1), "Stop input must be a 1D tensor. Got: ", stop_rank);
    NODE_VALIDATION_CHECK(this, step_rank.compatible(1), "Step input must be a 1D tensor. Got: ", step_rank);

    const auto inputs_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          inputs_size == 4 || inputs_size == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          inputs_size);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    const auto start_const = get_constant_from_source(input_value(1));
    const auto stop_const = get_constant_from_source(input_value(2));
    const auto step_const = get_constant_from_source(input_value(3));

    std::shared_ptr<ngraph::op::v0::Constant> axes_const;
    if (get_input_size() > 4) {
        set_input_is_relevant_to_shape(4);
        axes_const = get_constant_from_source(input_value(4));
    } else {
        axes_const = get_default_const_axes(input_value(1));
    }

    if (start_const && stop_const && step_const && axes_const) {
        const std::vector<int64_t> starts = start_const->cast_vector<int64_t>();
        const std::vector<int64_t> stops = stop_const->cast_vector<int64_t>();
        const std::vector<int64_t> steps = step_const->cast_vector<int64_t>();
        const std::vector<int64_t> axes = axes_const->cast_vector<int64_t>();

        for (size_t i = 0; i < data_shape.rank().get_length(); ++i) {
            // Dynamic data_shape rank was handled on the begining
            const auto norm_axis = ngraph::normalize_axis(this, axes[i], data_shape.rank());

            auto start = starts[i];
            auto stop = stops[i];
            auto step = steps[i];

            NODE_VALIDATION_CHECK(this, step != 0, "'step' value can't be zero!");

            const auto& axis_dim = data_shape[norm_axis];
            if (axis_dim.is_dynamic()) {
                if (start < 0 || stop < 0) {  // Can't be normalized
                    output_shape[norm_axis] = Dimension(0, axis_dim.get_max_length());
                    continue;
                }
            }

            const auto axis_dim_length = axis_dim.get_max_length();

            // Normalize indices
            start = start < 0 ? axis_dim_length + start : start;
            stop = stop < 0 ? axis_dim_length + stop : stop;

            // Check bounds intersection
            const bool no_intersection_negative_step = (start <= stop || start < 0) && step < 0;
            const bool no_intersection_positive_step = (start >= axis_dim_length || start >= stop) && step > 0;
            if (no_intersection_positive_step || no_intersection_negative_step) {
                output_shape[norm_axis] = 0;
                continue;
            }

            // Clip bounds values according to the dim size
            start = std::max(int64_t(0), std::min(start, axis_dim_length - 1));  // inclusive
            stop = std::max(int64_t(-1), std::min(stop, axis_dim_length));       // exclusive

            const auto elements_in_range = std::ceil(std::fabs(stop - start) / fabs(step));
            output_shape[norm_axis] = elements_in_range;
        }
    } else {
        if (axes_const) {
            // If we know only axes values, we should update lower_bound to 0 value,
            // for the specified dims by the axes. For unspecified dims, bounds as in data_shape.
            for (const auto& axis : axes_const->cast_vector<int64_t>()) {
                if (axis < data_shape.rank().get_length()) {
                    output_shape[axis] = Dimension(0, data_shape[axis].get_max_length());
                }
                // TODO: else throw, or check the values in advance
            }
        } else {
            // Otherwise axes values are also unknown,
            // then all of the output dims can be 0, so lower_bound = 0.
            for (size_t i = 0; i < data_shape.rank().get_length(); ++i) {
                output_shape[i] = Dimension(0, data_shape[i].get_max_length());
            }
        }
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::v8::Slice::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v8_Slice_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<v8::Slice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    } else {
        return std::make_shared<v8::Slice>(new_args.at(0),
                                           new_args.at(1),
                                           new_args.at(2),
                                           new_args.at(3),
                                           new_args.at(4));
    }
}
