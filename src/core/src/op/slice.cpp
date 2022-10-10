// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/slice.hpp"

#include <numeric>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v8::Slice);

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
    const int64_t rest = elements_in_range % std::abs(step);
    const int64_t integer_div = elements_in_range / std::abs(step);
    const int64_t sliced_dim_size = !rest ? integer_div : integer_div + 1;
    return sliced_dim_size;
}

bool is_max_int(element::Type_t ind_type, int64_t value) {
    int64_t max_type_value = 0;
    switch (ind_type) {
    case element::i32:
        max_type_value = std::numeric_limits<typename element_type_traits<element::i32>::value_type>::max();
        break;
    case element::i64:
        max_type_value = std::numeric_limits<typename element_type_traits<element::i64>::value_type>::max();
        break;
    default:
        return false;
    }
    return max_type_value == value;
}
}  // namespace

bool op::v8::Slice::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_Slice_visit_attributes);
    return true;
}

std::shared_ptr<ngraph::op::v0::Constant> op::v8::Slice::get_default_const_axes(const Output<Node>& start) const {
    const auto start_pshape = start.get_partial_shape();
    // Static case
    if (start_pshape.rank().is_static() && start_pshape.rank().get_length() == 1 && start_pshape[0].is_static()) {
        size_t axes_length = start_pshape[0].get_length();
        std::vector<int64_t> axes(axes_length);
        std::iota(axes.begin(), axes.end(), 0);
        return op::v0::Constant::create(element::i64, Shape{axes_length}, axes);
    }
    // Dynamic case
    return nullptr;
}

void op::v8::Slice::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Slice_validate_and_infer_types);

    const auto inputs_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          inputs_size == 4 || inputs_size == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          inputs_size);

    const PartialShape& data_shape = get_input_partial_shape(0);
    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          data_rank.is_dynamic() || data_rank.get_length() > 0,
                          "Slice `data` input can't be a scalar.");

    const auto start_const = get_constant_from_source(input_value(1));
    const auto stop_const = get_constant_from_source(input_value(2));
    const auto step_const = get_constant_from_source(input_value(3));

    const auto& start_input = start_const ? start_const : input_value(1);
    const auto& stop_input = stop_const ? stop_const : input_value(2);
    const auto& step_input = step_const ? step_const : input_value(3);

    NODE_VALIDATION_CHECK(this,
                          start_input.get_element_type().is_integral_number(),
                          "Slice `start` input type must be integer.");
    NODE_VALIDATION_CHECK(this,
                          stop_input.get_element_type().is_integral_number(),
                          "Slice `stop` input type must be integer.");
    NODE_VALIDATION_CHECK(this,
                          step_input.get_element_type().is_integral_number(),
                          "Slice `step` input type must be integer.");

    const auto& start_shape = start_input.get_partial_shape();
    const auto& stop_shape = stop_input.get_partial_shape();
    const auto& step_shape = step_input.get_partial_shape();

    const auto& start_rank = start_shape.rank();
    const auto& stop_rank = stop_shape.rank();
    const auto& step_rank = step_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          start_rank.compatible(1),
                          "Slice `start` input must be a 1D tensor. Got rank: ",
                          start_rank);
    NODE_VALIDATION_CHECK(this,
                          stop_rank.compatible(1),
                          "Slice `stop` input must be a 1D tensor. Got rank: ",
                          stop_rank);
    NODE_VALIDATION_CHECK(this,
                          step_rank.compatible(1),
                          "Slice `step` input must be a 1D tensor. Got rank: ",
                          step_rank);

    if (data_rank.is_static()) {
        const auto data_rank_length = data_rank.get_length();
        NODE_VALIDATION_CHECK(this,
                              start_rank.is_dynamic() || start_shape[0].get_min_length() <= data_rank_length,
                              "Slice `start` input dim size can't be bigger than `data` rank.");
        NODE_VALIDATION_CHECK(this,
                              stop_rank.is_dynamic() || stop_shape[0].get_min_length() <= data_rank_length,
                              "Slice `stop` input dim size can't be bigger than `data` rank.");
        NODE_VALIDATION_CHECK(this,
                              step_rank.is_dynamic() || step_shape[0].get_min_length() <= data_rank_length,
                              "Slice `step` input dim size can't be bigger than `data` rank.");
    }

    NODE_VALIDATION_CHECK(
        this,
        start_shape.compatible(stop_shape) && start_shape.compatible(step_shape) && stop_shape.compatible(step_shape),
        "Slice `start`, `stop`, `step` inputs must have compatible shapes.");

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    std::shared_ptr<ngraph::op::v0::Constant> axes_const;
    if (get_input_size() > 4) {
        set_input_is_relevant_to_shape(4);
        axes_const = get_constant_from_source(input_value(4));
        const auto& axes_input = axes_const ? axes_const : input_value(4);
        const auto& axes_rank = axes_input.get_partial_shape().rank();
        NODE_VALIDATION_CHECK(this,
                              axes_rank.compatible(1),
                              "Slice `axes` input must be a 1D tensor. Got rank: ",
                              axes_rank);
        NODE_VALIDATION_CHECK(this,
                              axes_rank.is_dynamic() || axes_input.get_partial_shape()[0].get_max_length() <=
                                                            data_rank.get_interval().get_max_val(),
                              "Slice `axes` input dim size can't be bigger than `data` rank.");
        NODE_VALIDATION_CHECK(this,
                              axes_input.get_partial_shape().compatible(start_shape),
                              "Slice `axes` input must have compatible shape with `start`, `stop`, `step` inputs.");
        NODE_VALIDATION_CHECK(this,
                              axes_input.get_element_type().is_integral_number(),
                              "Slice `axes` input type must be integer.");
    } else {
        axes_const = get_default_const_axes(start_input);
    }

    PartialShape output_shape(data_shape);

    // If data_shape rank is dynamic we can't calulate output shape.
    // Even with const start/stop/step/axes, we don't know how many axes should be copied
    // as "unspefified" in the final output shape, so the output shape rank is also dynamic.
    if (data_rank.is_dynamic()) {
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    if (start_const && stop_const && step_const && axes_const) {
        const auto& starts = start_const->cast_vector<int64_t>();
        const auto& stops = stop_const->cast_vector<int64_t>();
        const auto& steps = step_const->cast_vector<int64_t>();
        const auto& axes = axes_const->cast_vector<int64_t>();

        output_shape = calculate_output_shape(starts, stops, steps, axes, data_shape);
    } else {
        const auto data_static_rank = data_shape.rank().get_length();
        OPENVINO_ASSERT(data_static_rank >= 0);
        if (axes_const) {
            // If we know only `axes` values, we should update lower_bound to 0 value,
            // for the specified dims by the axes. For unspecified dims, bounds as in data_shape.
            for (const auto& axis : axes_const->cast_vector<int64_t>()) {
                const auto norm_axis = axis < 0 ? data_static_rank + axis : axis;
                NODE_VALIDATION_CHECK(this,
                                      norm_axis >= 0 && norm_axis < data_static_rank,
                                      "Values in the `axes` input must be in range of the `data` input rank: [-",
                                      data_static_rank,
                                      ", ",
                                      data_static_rank - 1,
                                      "]. Got: ",
                                      axis);
                output_shape[norm_axis] = Dimension(0, data_shape[norm_axis].get_max_length());
            }
        } else {
            // Otherwise `axes` values are also unknown,
            // then all of the output dims can be 0, so have lower bound = 0.
            for (size_t i = 0; i < static_cast<size_t>(data_static_rank); ++i) {
                output_shape[i] = Dimension(0, data_shape[i].get_max_length());
            }
        }
    }
    set_output_type(0, get_input_element_type(0), output_shape);
}

std::shared_ptr<Node> op::v8::Slice::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Slice_clone_with_new_inputs);
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

PartialShape op::v8::Slice::calculate_output_shape(const std::vector<int64_t>& starts,
                                                   const std::vector<int64_t>& stops,
                                                   const std::vector<int64_t>& steps,
                                                   const std::vector<int64_t>& axes,
                                                   const PartialShape& data_shape) const {
    OV_OP_SCOPE(v8_Slice_calculate_output_shape);
    const auto ind_size = starts.size();
    NODE_VALIDATION_CHECK(this,
                          stops.size() == ind_size && steps.size() == ind_size && axes.size() == ind_size,
                          "Slice `start`, `stop`, `step`, `axes` inputs need to have the same size.");

    std::unordered_set<int64_t> axes_set(axes.begin(), axes.end());
    NODE_VALIDATION_CHECK(this, axes_set.size() == axes.size(), "Slice values in `axes` input must be unique.");

    PartialShape output_shape(data_shape);
    if (data_shape.rank().is_dynamic()) {
        return output_shape;
    }

    const auto data_static_rank = data_shape.rank().get_length();
    for (size_t i = 0; i < axes.size(); ++i) {
        const auto norm_axis = axes[i] < 0 ? data_static_rank + axes[i] : axes[i];
        NODE_VALIDATION_CHECK(this,
                              norm_axis >= 0 && norm_axis < data_static_rank,
                              "Values in the `axes` input must be in range of the `data` input rank: [-",
                              data_static_rank,
                              ", ",
                              data_static_rank - 1,
                              "]. Got: ",
                              axes[i]);

        auto start = starts[i];
        auto stop = stops[i];
        auto step = steps[i];

        NODE_VALIDATION_CHECK(this, step != 0, "Slice 'step' value can't be zero.");

        const auto& axis_dim = data_shape[norm_axis];
        const auto axis_min_dim_length = axis_dim.get_min_length();
        const auto min_dim_size = get_sliced_dim_size(start, stop, step, axis_min_dim_length);
        if (axis_dim.is_static()) {
            output_shape[norm_axis] = min_dim_size;
            continue;
        }

        // Avoid negative index normalization without upper bounds
        if (!axis_dim.get_interval().has_upper_bound()) {
            if (is_max_int(get_input_element_type(2), stop) || is_max_int(get_input_element_type(1), start)) {
                output_shape[norm_axis] = Dimension(-1);
                continue;
            } else if ((step < 0 && start < 0 && stop > 0) || (step > 0 && stop < 0 && start >= 0)) {
                output_shape[norm_axis] = Dimension(-1);
                continue;
            } else if (step < 0 && start > 0 && stop < 0) {
                output_shape[norm_axis] = Dimension(0, start + 1);
                continue;
            } else if (step > 0 && stop > 0 && start < 0) {
                output_shape[norm_axis] = Dimension(0, stop);
                continue;
            }
        }

        // Calculate max dim length (upper bound)
        auto axis_max_dim_length = axis_dim.get_interval().get_max_val();
        const auto max_dim_size = get_sliced_dim_size(start, stop, step, axis_max_dim_length);
        output_shape[norm_axis] = Dimension(min_dim_size, max_dim_size);
    }
    return output_shape;
}

bool op::v8::Slice::has_evaluate() const {
    OV_OP_SCOPE(v8_Slice_has_evaluate);
    switch (get_input_element_type(1)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        break;
    default:
        return false;
    }

    if (get_input_size() > 4) {
        switch (get_input_element_type(4)) {
        case ngraph::element::i8:
        case ngraph::element::i16:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u8:
        case ngraph::element::u16:
        case ngraph::element::u32:
        case ngraph::element::u64:
            break;
        default:
            return false;
        }
    }

    return true;
}

bool op::v8::Slice::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v8_Slice_evaluate);

    OPENVINO_ASSERT(inputs.size() >= 4, "Slice evaluate needs at least 4 inputs.");
    std::vector<int64_t> starts = host_tensor_2_vector<int64_t>(inputs[1]);
    std::vector<int64_t> stops = host_tensor_2_vector<int64_t>(inputs[2]);
    std::vector<int64_t> steps = host_tensor_2_vector<int64_t>(inputs[3]);

    std::vector<int64_t> axes(starts.size());
    if (inputs.size() < 5) {
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        axes = host_tensor_2_vector<int64_t>(inputs[4]);
    }

    // Static HostTensor data shape is needed to clamp and normalize `start` values
    const auto& data_shape = inputs[0]->get_partial_shape();
    OPENVINO_ASSERT(data_shape.is_static(), "Can't evaluate Slice elements without static HostTensor data shape.");
    // We need calculate static output shape based on HostTensor inputs
    PartialShape output_shape = calculate_output_shape(starts, stops, steps, axes, data_shape);
    OPENVINO_ASSERT(output_shape.is_static(), "Can't calculate static output shape for Slice evaluation.");

    outputs[0]->set_shape(output_shape.to_shape());
    outputs[0]->set_element_type(inputs[0]->get_element_type());

    ngraph::runtime::reference::slice(inputs[0]->get_data_ptr<char>(),
                                      data_shape.to_shape(),
                                      outputs[0]->get_data_ptr<char>(),
                                      output_shape.to_shape(),
                                      inputs[0]->get_element_type().size(),
                                      starts,
                                      steps,
                                      axes);
    return true;
}

namespace {
bool slice_input_check(const ov::Node* node) {
    if (!node->get_input_tensor(1).has_and_set_bound())
        return false;
    if (!node->get_input_tensor(2).has_and_set_bound())
        return false;
    if (!node->get_input_tensor(3).has_and_set_bound())
        return false;
    if (node->get_input_size() == 5 && !node->get_input_tensor(4).has_and_set_bound())
        return false;
    return true;
}
}  // namespace

bool op::v8::Slice::evaluate_lower(const HostTensorVector& output_values) const {
    if (!slice_input_check(this))
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v8::Slice::evaluate_upper(const HostTensorVector& output_values) const {
    if (!slice_input_check(this))
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v8::Slice::evaluate_label(TensorLabelVector& output_labels) const {
    if (!slice_input_check(this))
        return false;
    return default_label_evaluator(this, output_labels);
}
