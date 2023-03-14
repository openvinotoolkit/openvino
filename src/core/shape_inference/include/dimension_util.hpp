// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "openvino/util/common_util.hpp"

namespace ov {
namespace util {
namespace dim {

constexpr auto inf_bound = -1;  //!< Infinite bound value for dimension.

/**
 * @brief Calculate dilated dimension value.
 *
 * @param dim       Dimension size value.
 * @param dilation  Dilation value
 * @return          Dilated dimension value.
 */
template <class T>
constexpr auto dilated(const T dim, const T dilation) -> T {
    return (dim < 1) ? inf_bound : dilation * (dim - 1) + 1;
}

/**
 * @brief Calculate padded dimension size as dim size + padding size
 *
 * @param dim      Dimension size value.
 * @param pad_num  Number of padded dimension.
 * @return         Padded dimension value or infinite bound.
 */
constexpr auto padded(const int64_t dim, const int64_t pad_num) -> int64_t {
    return ((dim == inf_bound) || (dim + pad_num < 0)) ? inf_bound : dim + pad_num;
}

/**
 * @brief Divide dimension using ceil rounding.
 *
 * @tparam TDim    Dimension type.
 * @tparam T       Dimension length value type.
 *
 * @param dim      Input dimension.
 * @param divisor  Dimension division.
 * @return Divided dimension with bounds round up.
 */
template <class TDim, class T = typename TDim::value_type>
auto ceil_div(const TDim& dim, const T divisor) -> TDim {
    if (dim.is_static()) {
        return {util::ceil_div<T>(dim.get_length(), divisor)};
    } else if (dim.get_max_length() == static_cast<T>(dim::inf_bound)) {
        return {dim};
    } else {
        return {util::ceil_div<T>(dim.get_min_length(), divisor), util::ceil_div<T>(dim.get_max_length(), divisor)};
    }
}

/**
 * @brief Divide dimension using floor rounding.
 *
 * @tparam TDim    Dimension type.
 * @tparam T       Dimension length value type.
 *
 * @param dim      Input dimension.
 * @param divisor  Dimension division.
 * @return Divided dimension with bound round down.
 */
template <class TDim, class T = typename TDim::value_type>
auto floor_div(const TDim& dim, const T divisor) -> TDim {
    if (dim.is_static()) {
        return {dim.get_length() / divisor};
    } else if (dim.get_max_length() == static_cast<T>(dim::inf_bound)) {
        return {dim};
    } else {
        return {dim.get_min_length() / divisor, dim.get_max_length() / divisor};
    }
}

}  // namespace dim
}  // namespace util
}  // namespace ov
