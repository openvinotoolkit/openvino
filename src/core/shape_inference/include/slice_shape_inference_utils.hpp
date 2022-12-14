// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/op/constant.hpp>

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
    const auto is_norm_dim_max = ov::internal::is_max(norm_dim);
    const int64_t lower_max = is_reverse_step ? norm_dim - 1 : norm_dim;
    const int64_t upper_min = is_reverse_step ? inf_bound : min_bound;

    const auto is_start_lt_min_bound = start < min_bound;
    const auto are_bounds_diff_sign = is_start_lt_min_bound != (stop < 0);

    const auto is_start_max = ov::internal::is_max(start);
    const auto is_start_limit = is_start_max || ov::internal::is_min(start);
    const auto any_bound_max = is_start_max || ov::internal::is_max(stop);
    // Prepare bounds for number of elements calculation.
    int64_t lb, ub;
    if (is_norm_dim_max && (are_bounds_diff_sign || any_bound_max || is_start_limit)) {
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

    // std::cout << "for " << norm_dim << " [" << lb << ":" << ub << "]" << std::endl;

    // Calculate elements in step from bounds and step.
    if (is_norm_dim_max && lb == min_bound && ub == inf_bound) {
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

/**
 * \brief Get the input bounds from constants maps or evaluate bunds
 *  and return them as pair of vector (lower, upper)
 *
 * \tparam TShape        Shape type,
 *
 * \param op             Operator pointer.
 * \param idx            Input index.
 * \param constant_data  Map with constant data.
 *
 * \return Return pairs of vector.
 */
template <class TShape>
std::pair<std::vector<int64_t>, std::vector<int64_t>> get_input_bounds(
    const ov::Node* op,
    size_t idx,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    std::vector<int64_t> lower, upper;
    if (!get_data_as_int64<TShape>(idx, op, lower, constant_data)) {
        // if no const data try get input bounds
        auto bounds = ngraph::evaluate_both_bounds(op->get_input_source_output(idx));

        if (bounds.first && bounds.second) {
            lower = std::make_shared<op::v0::Constant>(bounds.first)->cast_vector<int64_t>();
            upper = std::make_shared<op::v0::Constant>(bounds.second)->cast_vector<int64_t>();
        }
    }

    return std::make_pair(lower, upper);
}
}  // namespace slice
}  // namespace op
}  // namespace ov
