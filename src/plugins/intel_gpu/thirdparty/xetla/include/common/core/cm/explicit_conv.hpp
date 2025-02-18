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
#include "../../../common/core/cm/base_ops.hpp"
#include "../../../common/core/cm/base_types.hpp"
#include "../../../common/core/cm/common.hpp"
#include "../../../common/core/cm/math_general.hpp"
#else
#include "common/core/cm/base_ops.hpp"
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/common.hpp"
#include "common/core/cm/math_general.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_conv
/// @{

//template <typename T_dst, typename T_src, int N>
//__XETLA_API xetla_vector<T_dst, N> xetla_cvt(xetla_vector<T_src, N> src) {}

/// @brief xetla explicit data conversion for standard data types(integer,float,half)
/// @tparam T_dst is the destination data type.
/// @tparam T_src is the source data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<!(is_internal_type<T_dst>::value)
                && !(is_internal_type<T_src>::value),
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
    xetla_vector<T_dst, N> dst = src;
    return dst;
}

/// @brief xetla explicit data conversion, fp32->bf16.
/// @tparam T_dst is the float32 data type.
/// @tparam T_src is the bfloat16 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, bf16>::value
                && std::is_same<T_src, float>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
    return cm_bf_cvt<half, T_src, N>(src).xetla_format<native_type_t<T_dst>>();
}

/// @brief xetla explicit data conversion, bf16->fp32.
/// @tparam T_dst is the bfloat16 data type.
/// @tparam T_src is the float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, float>::value
                && std::is_same<T_src, bf16>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
    return cm_bf_cvt<T_dst, half, N>(src.xetla_format<half>());
}

/// @brief xetla explicit data conversion, fp32->tf32.
/// @tparam T_dst is the float32 data type.
/// @tparam T_src is the tensor_float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, tf32>::value
                && std::is_same<T_src, float>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
    return cm_tf32_cvt<native_type_t<T_dst>, T_src, N>(src);
}

/// @brief xetla explicit data conversion, tf32->fp32.
/// @tparam T_dst is the tensor_float32 data type.
/// @tparam T_src is the float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, float>::value
                && std::is_same<T_src, tf32>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
    return src.xetla_format<float>();
}

/// @brief xetpp explicit data conversion with scaling, int32->fp16.
/// @tparam T_dst is the half data type.
/// @tparam T_src is the int32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, fp16>::value
                && std::is_same<T_src, int32_t>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src, float scaling_value) {
    xetla_vector<T_dst, N> dst = scaling_value * src;
    return dst;
}

/// @brief xetpp explicit data conversion with re-quantization, int32->int8.
/// @tparam T_dst is the int32 data type.
/// @tparam T_src is the int8 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, int8_t>::value
                && std::is_same<T_src, int32_t>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src, float scaling_value) {
    auto dst = xetla_vector<T_dst, N>(
            xetla_rnde<float>(scaling_value * src), SAT);
    return dst;
}

/// @brief xetpp explicit data conversion with scaling and quantization, float32->int8.
/// @tparam T_dst is the int8 data type.
/// @tparam T_src is the float32 data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, int8_t>::value
                && std::is_same<T_src, float>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src, float scaling_value) {
    auto dst = xetla_vector<T_dst, N>(
            xetla_rnde<float>(scaling_value * src), SAT);
    return dst;
}

/// @brief xetla explicit data conversion, same type.
/// @tparam T_dst is the dst data type.
/// @tparam T_src is the src data type.
/// @tparam N is the element number in xetla_vector.
template <typename T_dst, typename T_src, int N>
__XETLA_API typename std::enable_if_t<std::is_same<T_dst, T_src>::value
                && is_internal_type<T_src>::value,
        xetla_vector<T_dst, N>>
xetla_cvt(xetla_vector<T_src, N> src) {
    return src;
}

/// @} xetla_core_conv

} // namespace gpu::xetla
