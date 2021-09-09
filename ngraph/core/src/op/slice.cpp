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

OPENVINO_RTTI_DEFINITION(op::v8::Slice, "Slice", 8);

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
    } else  // Dynamic case
        return nullptr;
}

int64_t get_sliced_dim_size(int64_t start, int64_t stop, int64_t step, int64_t dim_size) {
    // Normalize index
    start = start < 0 ? dim_size + start : start;
    stop = stop < 0 ? dim_size + stop : stop;

    // Clip normalized bounds according to the dim size
    start = std::max(int64_t(0), std::min(start, dim_size));  // inclusive
    stop = std::max(int64_t(-1), std::min(stop, dim_size));   // exclusive

    int64_t elements_in_range = 0;
    if (step < 0) {
        // Clip max start index (last element inclusively)
        elements_in_range = std::max(int64_t(0), std::min(dim_size - 1, start) - stop);
    } else {
        // Clip max stop index (last element exclusively)
        elements_in_range = std::max(int64_t(0), std::min(dim_size, stop) - start);
    }
    const int64_t sliced_dim_size = std::ceil(elements_in_range / std::fabs(step));
    return sliced_dim_size;
}

}  // namespace

bool op::v8::Slice::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_Slice_visit_attributes);
    return true;
}

void op::v8::Slice::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v8_Slice_validate_and_infer_types);

    const auto inputs_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          inputs_size == 4 || inputs_size == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          inputs_size);

    const PartialShape& data_shape = get_input_partial_shape(0);
    PartialShape output_shape(data_shape);

    const auto& start_rank = get_input_partial_shape(1).rank();
    const auto& stop_rank = get_input_partial_shape(2).rank();
    const auto& step_rank = get_input_partial_shape(3).rank();

    NODE_VALIDATION_CHECK(this, start_rank.compatible(1), "Start input must be a 1D tensor. Got: ", start_rank);
    NODE_VALIDATION_CHECK(this, stop_rank.compatible(1), "Stop input must be a 1D tensor. Got: ", stop_rank);
    NODE_VALIDATION_CHECK(this, step_rank.compatible(1), "Step input must be a 1D tensor. Got: ", step_rank);

    const auto& start_input = input_value(1);
    const auto& stop_input = input_value(2);
    const auto& step_input = input_value(3);

    NODE_VALIDATION_CHECK(this,
                          start_input.get_element_type().is_integral_number(),
                          "Slice `start` input type must be integral.");
    NODE_VALIDATION_CHECK(this,
                          stop_input.get_element_type().is_integral_number(),
                          "Slice `stop` input type must be integral.");
    NODE_VALIDATION_CHECK(this,
                          step_input.get_element_type().is_integral_number(),
                          "Slice `step` input type must be integral.");

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    const auto start_const = get_constant_from_source(start_input);
    const auto stop_const = get_constant_from_source(stop_input);
    const auto step_const = get_constant_from_source(step_input);

    std::shared_ptr<ngraph::op::v0::Constant> axes_const;
    if (get_input_size() > 4) {
        set_input_is_relevant_to_shape(4);
        axes_const = get_constant_from_source(input_value(4));
    } else {
        axes_const = get_default_const_axes(input_value(1));
    }

    // If data_shape rank is dynamic we can't calulate output shape.
    // Even with const start/stop/step/axes, we don't know how many axes should be copied
    // as "unspefified" in the final output shape, so the output shape rank is also dynamic.
    if (data_shape.rank().is_dynamic()) {
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    if (start_const && stop_const && step_const && axes_const) {
        const std::vector<int64_t> starts = start_const->cast_vector<int64_t>();
        const std::vector<int64_t> stops = stop_const->cast_vector<int64_t>();
        const std::vector<int64_t> steps = step_const->cast_vector<int64_t>();
        const std::vector<int64_t> axes = axes_const->cast_vector<int64_t>();

        std::unordered_set<int64_t> axes_set(axes.begin(), axes.end());
        NODE_VALIDATION_CHECK(this, axes_set.size() == axes.size(), "Slice values in `axes` input must be unique.");

        for (size_t i = 0; i < starts.size(); ++i) {
            const auto norm_axis = ngraph::normalize_axis(this, axes[i], data_shape.rank());

            auto start = starts[i];
            auto stop = stops[i];
            auto step = steps[i];

            NODE_VALIDATION_CHECK(this, step != 0, "Sllice 'step' value can't be zero.");

            const auto& axis_dim = data_shape[norm_axis];
            const auto axis_min_dim_length = axis_dim.get_min_length();
            const auto min_dim_size = get_sliced_dim_size(start, stop, step, axis_min_dim_length);
            if (axis_dim.is_static()) {
                output_shape[norm_axis] = min_dim_size;
            }

            // Avoid negative index normalization without upper bounds
            if (!axis_dim.get_interval().has_upper_bound()) {
                if ((step < 0 && start < 0 && stop > 0) || (step > 0 && stop < 0 && start > 0)) {
                    output_shape[norm_axis] = Dimension(-1);
                    continue;
                } else if (step < 0 && start > 0 && stop < 0) {
                    int64_t max_out_dim = start >= INT32_MAX ? INT64_MAX : start + 1;
                    output_shape[norm_axis] = Dimension(0, max_out_dim);
                    continue;
                } else if (step > 0 && stop > 0 && start < 0) {
                    int64_t max_out_dim = stop >= INT32_MAX ? INT64_MAX : stop;
                    output_shape[norm_axis] = Dimension(0, max_out_dim);
                    continue;
                }
            }

            // Calculate max dim length (upper bound)
            auto axis_max_dim_length = axis_dim.get_interval().get_max_val();
            const auto max_dim_size = get_sliced_dim_size(start, stop, step, axis_max_dim_length);
            output_shape[norm_axis] = Dimension(min_dim_size, max_dim_size);
        }
    } else {
        if (axes_const) {
            // If we know only axes values, we should update lower_bound to 0 value,
            // for the specified dims by the axes. For unspecified dims, bounds as in data_shape.
            for (const auto& axis : axes_const->cast_vector<int64_t>()) {
                NODE_VALIDATION_CHECK(this,
                                      axis < data_shape.rank().get_length(),
                                      "Provided axis value must be in range of the data input rank. Got: ",
                                      axis);
                output_shape[axis] = Dimension(0, data_shape[axis].get_max_length());
            }
        } else {
            // Otherwise axes values are also unknown,
            // then all of the output dims can be 0, so have lower bound = 0.
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
