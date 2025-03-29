/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/utils/cm/common.hpp"
#include "../../../common/utils/cm/nd_item.hpp"
#else
#include "common/utils/cm/common.hpp"
#include "common/utils/cm/nd_item.hpp"
#endif

__XETLA_API constexpr uint32_t div_round_up(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

__XETLA_API constexpr int div_round_down(int n, int d) {
    // Numbers are rounded down towards the next lowest number
    // e.g. -2.0/3.0 ~ -0.666 -> -1.
    return (n - (((n % d) + d) % d)) / d;
}

__XETLA_API constexpr int modulo(int n, int d) {
    // Calculates modulo based on definition that uses floored divison.
    // Result has the same sign as d.
    return (d + (n % d)) % d;
}

namespace gpu::xetla {

/// @addtogroup xetla_util_misc
/// @{

/// @brief Returns time stamp
///
/// @return vecotor <uint32_t, 4> contain time stamp
///
__XETLA_API xetla_vector<uint32_t, 4> get_time_stamp() {
    return cm_rdtsc();
}

///
/// @brief xetla_vector generation.
/// Commonly used to generate simd channel id.
///
/// @tparam Ty is type of xetla_vector generated. Only `int` type(or types that can be implicitly converted to int) is allowed.
/// @tparam N is the number of elements in xetla_vector generated.
/// @param InitVal [in] is the starting value of xetla_vector generation.
/// @param Step [in] is the step value between adjacent elements.
/// @return xetla_vector<Ty, N> [out] is the return vector.
///
template <typename Ty, int N>
__XETLA_API xetla_vector<Ty, N> xetla_vector_gen(int InitVal, int Step) {
    xetla_vector<Ty, N> tmp;
    cmtl::cm_vector_assign(tmp, InitVal, Step);
    return tmp;
}

template <uint32_t N>
__XETLA_API xetla_mask_int<N> xetla_mask_int_gen(uint32_t mask_val) {
    return mask_val;
}

template <typename dtype_acc, uint32_t N, uint32_t num_flag = 4,
        typename dtype_mask = uint8_t>
__XETLA_API xetla_vector<dtype_acc, N> drop_out(xetla_vector<dtype_acc, N> in,
        xetla_vector<dtype_mask, N> mask, dtype_acc scale) {
    xetla_vector<dtype_acc, N> out = in * scale;
    constexpr uint32_t unroll_size = num_flag * 16;
    SW_BARRIER();
#pragma unroll
    for (int i = 0; i < N / unroll_size; i++) {
        xetla_mask<unroll_size> mask_flag
                = mask.xetla_select<unroll_size, 1>(i * unroll_size) > 0;
        out.xetla_select<unroll_size, 1>(i * unroll_size)
                .xetla_merge(0, mask_flag);
    }
    if constexpr (N % unroll_size != 0) {
        constexpr uint32_t remain_len = N % unroll_size;
        constexpr uint32_t remain_start = N / unroll_size * unroll_size;
        xetla_mask<remain_len> mask_flag
                = mask.xetla_select<remain_len, 1>(remain_start) > 0;
        out.xetla_select<remain_len, 1>(remain_start).xetla_merge(0, mask_flag);
    }
    return out;
}

template <reduce_op reduce_kind, typename dtype, int size>
__XETLA_API typename std::enable_if_t<reduce_kind == reduce_op::sum,
        xetla_vector<dtype, size>>
reduce_helper(xetla_vector<dtype, size> a, xetla_vector<dtype, size> b) {
    return a + b;
}

template <reduce_op reduce_kind, typename dtype, int size>
__XETLA_API typename std::enable_if_t<reduce_kind == reduce_op::prod,
        xetla_vector<dtype, size>>
reduce_helper(xetla_vector<dtype, size> a, xetla_vector<dtype, size> b) {
    return a * b;
}

template <reduce_op reduce_kind, typename dtype, int size>
__XETLA_API typename std::enable_if_t<reduce_kind == reduce_op::max,
        xetla_vector<dtype, size>>
reduce_helper(xetla_vector<dtype, size> a, xetla_vector<dtype, size> b) {
    xetla_vector<dtype, size> out;
    xetla_mask<size> mask = a > b;
    out.xetla_merge(a, b, mask);
    return out;
}

template <reduce_op reduce_kind, typename dtype, int size>
__XETLA_API typename std::enable_if_t<reduce_kind == reduce_op::min,
        xetla_vector<dtype, size>>
reduce_helper(xetla_vector<dtype, size> a, xetla_vector<dtype, size> b) {
    xetla_vector<dtype, size> out;
    xetla_mask<size> mask = a < b;
    out.xetla_merge(a, b, mask);
    return out;
}

template <reduce_op reduce_kind, typename dtype, int N_x, int N_y>
__XETLA_API typename std::enable_if_t<N_y == 1, xetla_vector<dtype, N_x>>
recur_row_reduce(xetla_vector<dtype, N_x> in) {
    return in;
}
template <reduce_op reduce_kind, typename dtype, int N_x, int N_y>
__XETLA_API typename std::enable_if_t<(N_y > 1), xetla_vector<dtype, N_x>>
recur_row_reduce(xetla_vector<dtype, N_x * N_y> in) {
    static_assert(((N_y) & (N_y - 1)) == 0, "N_y should be power of 2");
    xetla_vector<dtype, N_x * N_y / 2> temp;
    temp = reduce_helper<reduce_kind, dtype, N_x * N_y / 2>(
            in.xetla_select<N_x * N_y / 2, 1>(0),
            in.xetla_select<N_x * N_y / 2, 1>(N_x * N_y / 2));

    return recur_row_reduce<reduce_kind, dtype, N_x, N_y / 2>(temp);
}

template <reduce_op reduce_kind, typename dtype, int N_x, int N_y>
__XETLA_API typename std::enable_if_t<N_x == 1, xetla_vector<dtype, N_y>>
recur_col_reduce(xetla_vector<dtype, N_y> in) {
    return in;
}
template <reduce_op reduce_kind, typename dtype, int N_x, int N_y>
__XETLA_API typename std::enable_if_t<(N_x > 1), xetla_vector<dtype, N_y>>
recur_col_reduce(xetla_vector<dtype, N_x * N_y> in) {
    static_assert(((N_x) & (N_x - 1)) == 0, "N_x should be power of 2");
    xetla_vector<dtype, N_x * N_y / 2> temp;
    auto in_2d = in.xetla_format<dtype, N_y, N_x>();
    temp = reduce_helper<reduce_kind, dtype, N_y * N_x / 2>(
            in_2d.xetla_select<N_y, 1, N_x / 2, 1>(0, 0),
            in_2d.xetla_select<N_y, 1, N_x / 2, 1>(0, N_x / 2));

    return recur_col_reduce<reduce_kind, dtype, N_x / 2, N_y>(temp);
}

/// @brief get linear group id of the last two dimensions.
/// @tparam item Is the given sycl::nd_item<3>.
__XETLA_API uint32_t get_2d_group_linear_id(sycl::nd_item<3> &item) {
    return item.get_group(2) + item.get_group(1) * item.get_group_range(2);
}
/// @} xetla_util_misc

} // namespace gpu::xetla
