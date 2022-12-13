// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/slice.hpp>

#include "sequnce_generator.hpp"
#include "utils.hpp"

namespace ov {
namespace internal {
/**
 * \brief Check if value of type T has got maximum value of type U.
 *
 * \tparam T     Input value type
 * \tparam U     Type to get its minimum for comparision. Default same as T.
 *
 * \param value  Input value.
 *
 * \return       True if input value has got maximum value of type U otherwise false.
 */
template <class T, class U = T>
constexpr bool is_max(const T& value) {
    return std::numeric_limits<U>::max() == value;
}

/**
 * \brief Check if value of type T has got minimum value of type U.
 *
 * \tparam T     Input value type.
 * \tparam U     Type to get its minimum for comparision. Default same as T.
 *
 * \param value  Input value.
 *
 * \return       True if input value has got minimum value of type U otherwise false.
 */
template <class T, class U = T>
constexpr bool is_min(const T& value) {
    return std::numeric_limits<U>::min() == value;
}
}  // namespace internal

namespace element {
/**
 * \brief  Check if value has got maximum value of ov::element::Type_t
 *
 * \tparam T     Input value type.
 *
 * \param type   ov::element type to get its maximum.
 * \param value  Input value for check.
 *
 * \return True if input value has got maximum number specified by ov::element type otherwise false.
 */
template <class T>
bool is_max_of(const element::Type_t& type, const T& value) {
    switch (type) {
    case element::i32:
        return internal::is_max<T, typename element_type_traits<element::i32>::value_type>(value);
    case element::i64:
        return internal::is_max<T, typename element_type_traits<element::i64>::value_type>(value);
    default:
        return false;
    }
}

/**
 * \brief  Check if value has got minimum value of ov::element::Type_t
 *
 * \tparam T     Input value type.
 *
 * \param type   ov::element type to get its minimum.
 * \param value  Input value for check.
 *
 * \return True if input value has got minimum number specified by ov::element type otherwise false.
 */
template <class T>
bool is_min_of(const element::Type_t type, const T& value) {
    switch (type) {
    case element::i32:
        return internal::is_min<T, typename element_type_traits<element::i32>::value_type>(value);
    case element::i64:
        return internal::is_min<T, typename element_type_traits<element::i64>::value_type>(value);
    default:
        return false;
    }
}

/**
 * \brief  Checks input value for element type maximum or minimum and return limit or value.
 *
 * \tparam T     Type of input value.
 * \tparam U     Type of return value. Default same as T.
 *
 * \param type   Type of ov::element::Type_t
 * \param value  Input value for check.
 *
 * \return If value is maximum or minimum get limit of U otherwise value as U.
 */
template <class T, class U = T>
U get_value_or_limit_of(const element::Type_t& type, const T& value) {
    if (is_min_of(type, value)) {
        return std::numeric_limits<U>::min();
    } else if (is_max_of(type, value)) {
        return std::numeric_limits<U>::max();
    } else {
        return static_cast<U>(value);
    }
}

}  // namespace element

namespace op {
namespace slice {

const std::array<std::string, 4> shape_names{"start", "stop", "step", "axes"};

/**
 * \brief Get number of elements in step for dimension size (lower/upper) by start, stop, step.
 *
 * \param dim    Dimension size or its bound (upper/lower)
 * \param start  Start in dimension.
 * \param stop   Stop in dimension.
 * \param step   Number elements taken in one step.
 *
 * \return -1 for infinite number otherwise [0..int64_max] for finit step.
 */
inline int64_t get_step_elements(const int64_t& dim, const int64_t& start, const int64_t& stop, const int64_t& step) {
    const auto is_reverse_step = step < 0;

    constexpr int64_t min_bound = 0;
    constexpr int64_t inf_bound = -1;

    const auto& norm_dim = dim == inf_bound ? std::numeric_limits<int64_t>::max() : dim;
    const int64_t lower_max = is_reverse_step ? norm_dim - 1 : norm_dim;
    const int64_t upper_min = is_reverse_step ? inf_bound : min_bound;

    const auto is_start_lt_min_bound = start < min_bound;
    const auto are_bounds_diff_sign = is_start_lt_min_bound != (stop < 0);

    const auto is_start_max = ov::internal::is_max(start);
    const auto is_start_limit = is_start_max || ov::internal::is_min(start);
    const auto any_bound_max = is_start_max || ov::internal::is_max(stop);
    // Prepare bounds for number of elements calculation.
    int64_t lb, ub;
    if (ov::internal::is_max(norm_dim) && (are_bounds_diff_sign || any_bound_max || is_start_limit)) {
        if (is_reverse_step) {
            ub = (is_start_lt_min_bound || any_bound_max) ? inf_bound : inf_bound - start;
        } else {
            ub = (is_start_lt_min_bound && !is_start_limit) ? stop : inf_bound;
        }
        lb = min_bound;
    } else {
        lb = clip(normalize(start, norm_dim), min_bound, lower_max);
        ub = clip(normalize(stop, norm_dim), upper_min, norm_dim);
    }

    // Calculate elements in step from bounds and step.
    if (lb == min_bound && ub == inf_bound) {
        return inf_bound;
    } else {
        // Limit elements count to not-positive for negative step or not-negative for positive step
        auto step_elements =
            is_reverse_step ? std::min<int64_t>(min_bound, (ub - lb)) : std::max<int64_t>(min_bound, (ub - lb));

        if (step == -1) {
            // Elements count is negative for negative step return opposite
            step_elements = -step_elements;
        } else if (step_elements != 0 && step != 1) {
            // Need to calculate elements in step. Depends on step direction reduce number element
            // in order to calculate elements in steps in one-step division (no modulo required)
            is_reverse_step ? ++step_elements : --step_elements;
            step_elements /= step;
            ++step_elements;
        } else {
            // There is no need for calculations as number of elements is 0 or step is 1.
        }
        return step_elements;
    }
}
}  // namespace slice
namespace v8 {

template <class T>
void shape_infer(const Slice* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    const auto& num_of_inputs = input_shapes.size();

    NODE_VALIDATION_CHECK(op,
                          num_of_inputs == 4 || num_of_inputs == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          num_of_inputs);
    NODE_VALIDATION_CHECK(op, output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();

    std::accumulate(input_shapes.begin() + 1, input_shapes.end(), 0, [&op, &input_rank](int i, const T& shape) -> int {
        const auto& shape_rank = shape.rank();
        NODE_VALIDATION_CHECK(op,
                              shape_rank.compatible(1),
                              "Slice `",
                              slice::shape_names[i],
                              "` input must be a 1D tensor. Got rank: ",
                              shape_rank);

        if (input_rank.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  shape_rank.is_dynamic() || shape[0].get_min_length() <= input_rank.get_length(),
                                  "Slice `",
                                  slice::shape_names[i],
                                  "` input dim size can't be bigger than `data` rank.");
        }
        return ++i;
    });

    const auto& start_shape = input_shapes[1];
    const auto& stop_shape = input_shapes[2];
    const auto& step_shape = input_shapes[3];

    NODE_VALIDATION_CHECK(
        op,
        start_shape.compatible(stop_shape) && start_shape.compatible(step_shape) && stop_shape.compatible(step_shape),
        "Slice `start`, `stop`, `step` inputs must have compatible shapes.");

    // it is not possible to define output shape if input data shape rank is undefined
    // even the lengths of begin, end, or strides are defined
    if (input_rank.is_dynamic()) {
        output_shapes[0] = PartialShape::dynamic();
        return;
    }

    std::vector<int64_t> start, stop, steps, axes;

    // compute constant values of begin, end, and strides if possible
    const auto got_start = get_data_as_int64<T>(1, op, start, constant_data);
    const auto got_stop = get_data_as_int64<T>(2, op, stop, constant_data);
    const auto got_steps = get_data_as_int64<T>(3, op, steps, constant_data);

    bool got_axes;
    if (input_shapes.size() > 4) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[4].compatible(start_shape),
                              "Slice `axes` input must have compatible shape with `start`, `stop`, `step` inputs.");
        got_axes = get_data_as_int64<T>(4, op, axes, constant_data);
        if (got_axes) {
            NODE_VALIDATION_CHECK(op, ov::are_unique(axes), "Slice values in `axes` input must be unique.");
            ov::normalize_axes(op, input_shape.rank().get_length(), axes);
        }
    } else if (got_start) {
        axes.reserve(start.size());
        std::generate_n(std::back_inserter(axes), start.size(), SeqGen<int64_t>(0));
        got_axes = true;
    } else {
        got_axes = false;
    }

    auto& output_shape = output_shapes[0];

    std::vector<DimType> dims;
    dims.reserve(input_shape.rank().get_length());
    for (size_t dim_idx = 0; dim_idx < input_shape.rank().get_length(); ++dim_idx) {
        const DimType& input_dim = input_shape[dim_idx];

        const auto axis_it = std::find(axes.begin(), axes.end(), dim_idx);
        if (axis_it != axes.end()) {
            const auto i = std::distance(axes.begin(), axis_it);

            if (got_start && got_stop && got_steps) {
                const auto& step = steps[i];
                NODE_VALIDATION_CHECK(op, step != 0, "Step must be non-zero");

                const auto& start_lb = element::get_value_or_limit_of<int64_t>(op->get_input_element_type(1), start[i]);
                auto& start_ub = start_lb;

                const auto& stop_lb = element::get_value_or_limit_of<int64_t>(op->get_input_element_type(1), stop[i]);
                auto& stop_ub = stop_lb;

                auto lb = slice::get_step_elements(input_dim.get_min_length(), start_ub, stop_lb, step);
                auto ub = slice::get_step_elements(input_dim.get_max_length(), start_lb, stop_ub, step);
                dims.emplace_back(lb, ub);
            } else {
                dims.emplace_back(0, input_dim.get_max_length());
            }

            if (std::is_same<DimType, ov::Dimension>::value && dims.back() == input_dim) {
                // for equal ov::Dimension do merge to get input label (always success)
                DimType::merge(dims.back(), dims.back(), input_dim);
            }
        } else {
            dims.push_back(input_dim);
        }
    }
    output_shape = T(std::move(dims));
}
}  // namespace v8
}  // namespace op
}  // namespace ov
