// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/slice.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
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

bool op::v8::Slice::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_Slice_visit_attributes);
    return true;
}

void op::v8::Slice::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v8_Slice_validate_and_infer_types);

    const PartialShape& data_shape = get_input_partial_shape(0);
    PartialShape output_shape(data_shape);

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

    if (get_input_size() < 5) {
        set_argument(4, calculate_default_axes(get_input_node_ptr(1)->output(0)));
    }

    set_input_is_relevant_to_shape(4);

    const auto start_const = get_constant_from_source(input_value(1));
    const auto stop_const = get_constant_from_source(input_value(2));
    const auto step_const = get_constant_from_source(input_value(3));
    const auto axes_const = get_constant_from_source(input_value(4));

    if (start_const && stop_const && step_const && axes_const) {
        const std::vector<int64_t> starts = start_const->cast_vector<int64_t>();
        const std::vector<int64_t> stops = stop_const->cast_vector<int64_t>();
        const std::vector<int64_t> steps = step_const->cast_vector<int64_t>();

        std::vector<int64_t> axis_vector = axes_const->cast_vector<int64_t>();

        for (auto axis : axis_vector) {
            const auto norm_axis = ngraph::normalize_axis(this, axis, data_shape.rank());

            auto start = starts[norm_axis];
            auto stop = stops[norm_axis];
            auto step = steps[norm_axis];

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
            output_shape[axis] = elements_in_range;
        }
    }

    // TODO: Dynamic case
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
