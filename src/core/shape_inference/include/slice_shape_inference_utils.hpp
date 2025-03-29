// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "sequence_generator.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace slice {

/**
 * \brief Get sliced value in step for given dimension value and start, stop, step.
 *
 * \note This function cannot be use for step 0 (division by 0)
 *
 * \param dim    Dimension value.
 * \param start  Start of slice.
 * \param stop   Stop of slice.
 * \param step   Step of slice.
 *
 * \return -1 for infinite number otherwise [0..int64_max] for finit step.
 */
inline int64_t get_sliced_value(const int64_t dim, const int64_t start, const int64_t stop, const int64_t step) {
    using namespace ov::util;

    const auto is_reverse_step = step < 0;

    constexpr int64_t min_bound = 0;

    const auto& norm_dim = dim::is_inf_bound(dim) ? std::numeric_limits<int64_t>::max() : dim;
    const auto is_norm_dim_max = ov::util::is_max(norm_dim);

    const auto is_start_lt_min_bound = start < min_bound;
    const auto are_bounds_diff_sign = is_start_lt_min_bound != (stop < 0);

    const auto is_start_max = ov::util::is_max(start);
    const auto is_start_limit = is_start_max || ov::util::is_min(start);
    const auto is_stop_max = ov::util::is_max(stop);
    const auto any_bound_max = is_start_max || is_stop_max;
    // Prepare bounds for sliced value calculation.
    int64_t lb, ub;
    if (is_norm_dim_max && (are_bounds_diff_sign || any_bound_max || is_start_limit)) {
        if (is_reverse_step) {
            ub = (is_start_lt_min_bound || any_bound_max) ? dim::inf_bound : dim::inf_bound - start;
        } else if (is_start_lt_min_bound && !is_start_limit) {
            ub = is_stop_max ? -start : stop;
        } else {
            ub = dim::inf_bound;
        }
        lb = min_bound;
    } else {
        const int64_t lower_max = is_reverse_step ? norm_dim - 1 : norm_dim;
        const int64_t upper_min = is_reverse_step ? dim::inf_bound : min_bound;

        lb = ov::util::clip(ov::util::normalize(start, norm_dim), min_bound, lower_max);
        ub = ov::util::clip(ov::util::normalize(stop, norm_dim), upper_min, norm_dim);
    }

    // Calculate sliced value from bounds and step.
    if (is_norm_dim_max && lb == min_bound && dim::is_inf_bound(ub)) {
        return dim::inf_bound;
    } else {
        // Limit sliced value to not-positive for negative step or not-negative for positive step
        auto sliced_value =
            is_reverse_step ? std::min<int64_t>(min_bound, (ub - lb)) : std::max<int64_t>(min_bound, (ub - lb));

        if (step == -1) {
            // Sliced value is negative for negative step return opposite
            sliced_value = -sliced_value;
        } else if (sliced_value != 0 && step != 1) {
            // Need to calculate sliced value for step. Depends on step direction reduce sliced value
            // in order to calculate it in one-step division (no modulo required)
            is_reverse_step ? ++sliced_value : --sliced_value;
            sliced_value /= step;
            ++sliced_value;
        } else {
            // There is no need for calculations as sliced value is 0 or step is 1.
        }
        return sliced_value;
    }
}

using Bounds = std::pair<int64_t, int64_t>;  //!< Alias to dimension bounds for slice.

/**
 * @brief Check if bounds can cross 0 value (rising edge).
 *
 * @param b Input interval bounds for check.
 * @return True if lower bound is negative and upper is not negative, otherwise false.
 */
constexpr bool is_bounds_zero_crossing(const Bounds b) {
    return (b.first < 0) && (b.second >= 0);
}

/**
 * @brief Check if lower bound is within dimension.
 *
 * Check valid only if bounds can cross zero value (lb is negative).
 *
 * @param lb Lower bound for check.
 * @param dim Dimension used to check lower bound.
 * @return True if lower bound is within dimension length, otherwise false.
 */
template <class TDim>
constexpr bool is_lb_within_dim(const int64_t lb, const TDim& dim) {
    return (static_cast<int64_t>(dim.get_max_length()) == ov::util::dim::inf_bound) || lb + dim.get_max_length() >= 0;
}

/**
 * @brief Check if upper bound is within dimension.
 *
 * Check valid only if bounds can cross zero value (up is not negative).
 *
 * @param ub Upper bound for check.
 * @param dim Dimension used to check upper bound.
 * @return True if upper bound is within dimension length, otherwise false.
 */
template <class TDim>
constexpr bool is_ub_within_dim(const int64_t ub, const TDim& dim) {
    return (static_cast<int64_t>(dim.get_max_length()) == ov::util::dim::inf_bound) ||
           cmp::lt(ub, dim.get_max_length());
}

/**
 * \brief Make sliced dimension for input dimension by step from start to stop bounds.
 *
 * \tparam TDim   Type of in/out dimension.
 *
 * \param dim    Input Dimension to slice.
 * \param start  Slice start bounds.
 * \param stop   Slice stop bounds.
 * \param step   Slice step.
 *
 * \return Dimension with upper/lower values set according slice inputs.
 */
template <class TDim>
TDim make_dim(const TDim& dim, const Bounds& start, const Bounds& stop, int64_t step) {
    using TDimVal = typename TDim::value_type;
    using namespace ov::util;

    const auto is_start_zero_crossing = is_bounds_zero_crossing(start);
    const auto start_lb = is_start_zero_crossing && is_lb_within_dim(start.first, dim) ? 0 : start.first;
    const auto start_ub = is_start_zero_crossing && is_ub_within_dim(start.second, dim) ? -1 : start.second;

    const auto is_stop_zero_crossing = is_bounds_zero_crossing(stop);
    const auto stop_lb = is_stop_zero_crossing && is_lb_within_dim(stop.first, dim) ? 0 : stop.first;
    const auto stop_ub = is_stop_zero_crossing && is_ub_within_dim(stop.second, dim) ? -1 : stop.second;

    TDimVal lb, ub;
    if (step > 0) {
        lb = static_cast<TDimVal>(get_sliced_value(dim::value_convert(dim.get_min_length()), start_ub, stop_lb, step));
        ub = static_cast<TDimVal>(get_sliced_value(dim::value_convert(dim.get_max_length()), start_lb, stop_ub, step));
    } else {
        lb = static_cast<TDimVal>(get_sliced_value(dim::value_convert(dim.get_min_length()), start_lb, stop_ub, step));
        ub = static_cast<TDimVal>(get_sliced_value(dim::value_convert(dim.get_max_length()), start_ub, stop_lb, step));
    }

    return {lb, ub};
}
}  // namespace slice
}  // namespace op
}  // namespace ov
