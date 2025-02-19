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
#include "../../../common/core/cm/base_types.hpp"
#include "../../../common/core/cm/common.hpp"
#else
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_bit_manipulation
/// @{

/// Shift left operation (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vector.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted left values.
template <typename T0, typename T1, int SZ, typename U,
        class Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T0, SZ> xetla_shl(
        xetla_vector<T1, SZ> src0, U src1, Sat sat = {}) {
    return cm_shl<T0, T1, SZ, U>(src0, src1, Sat::value);
}

/// Shift left operation (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted left value.
template <typename T0, typename T1, typename T2,
        class Sat = xetla_saturation_off_tag>
typename std::remove_const<T0>::type xetla_shl(T1 src0, T2 src1, Sat sat = {}) {
    return cm_shl<T0, T1, T2>(src0, src1, Sat::value);
}

/// Shift right operation (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vector.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted right values.
template <typename T0, typename T1, int SZ, typename U,
        class Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T0, SZ> xetla_shr(
        xetla_vector<T1, SZ> src0, U src1, Sat sat = {}) {
    return cm_shr<T0, T1, SZ, U>(src0, src1, Sat::value);
}

/// Shift right operation (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted right value.
template <typename T0, typename T1, typename T2,
        class Sat = xetla_saturation_off_tag>
__XETLA_API typename std::remove_const<T0>::type xetla_shr(
        T1 src0, T2 src1, Sat sat = {}) {
    return cm_shr<T0, T1, T2>(src0, src1, Sat::value);
}

/// Rotate left operation with two vector inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the vector with number of bit positions by which the elements of
/// the input vector \p src0 shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ>
__XETLA_API xetla_vector<T0, SZ> xetla_rol(
        xetla_vector<T1, SZ> src0, xetla_vector<T1, SZ> src1) {
    return cm_rol<T0, T1, SZ>(src0, src1);
}

/// Rotate left operation with a vector and a scalar inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ, typename U>
__XETLA_API std::enable_if_t<std::is_integral<T0>::value
                && std::is_integral<T1>::value && std::is_integral<U>::value
                && is_xetla_scalar<U>::value,
        xetla_vector<T0, SZ>>
xetla_rol(xetla_vector<T1, SZ> src0, U src1) {
    return cm_rol<T0, T1, SZ, U>(src0, src1);
}

/// Rotate left operation with two scalar inputs
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated left value.
template <typename T0, typename T1, typename T2>
__XETLA_API std::enable_if_t<std::is_integral<T0>::value
                && std::is_integral<T1>::value && std::is_integral<T2>::value,
        remove_const_t<T0>>
xetla_rol(T1 src0, T2 src1) {
    return cm_rol<T0, T1, T2>(src0, src1);
}

/// Rotate right operation with two vector inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the vector with number of bit positions by which the elements of
/// the input vector \p src0 shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ>
__XETLA_API xetla_vector<T0, SZ> xetla_ror(
        xetla_vector<T1, SZ> src0, xetla_vector<T1, SZ> src1) {
    return cm_ror<T0, T1, SZ>(src0, src1);
}

/// Rotate right operation with a vector and a scalar inputs
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return vector of rotated elements.
template <typename T0, typename T1, int SZ, typename U>
__XETLA_API std::enable_if_t<std::is_integral<T0>::value
                && std::is_integral<T1>::value && std::is_integral<U>::value
                && is_xetla_scalar<U>::value,
        xetla_vector<T0, SZ>>
xetla_ror(xetla_vector<T1, SZ> src0, U src1) {
    return cm_ror<T0, T1, SZ, U>(src0, src1);
}

/// Rotate right operation with two scalar inputs
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value. Must be any integer type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be rotated.
/// @return rotated right value.
template <typename T0, typename T1, typename T2>
__XETLA_API std::enable_if_t<std::is_integral<T0>::value
                && std::is_integral<T1>::value && std::is_integral<T2>::value,
        remove_const_t<T0>>
xetla_ror(T1 src0, T2 src1) {
    return cm_ror<T0, T1, T2>(src0, src1);
}

/// Logical Shift Right (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted elements.
template <typename T0, typename T1, int SZ, typename U,
        class Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T0, SZ> xetla_lsr(
        xetla_vector<T1, SZ> src0, U src1, Sat sat = {}) {
    return cm_lsr<T0, T1, SZ, U>(src0, src1, Sat::value);
}

/// Logical Shift Right (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value \p src0. Must be any integer
/// type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted value.
template <typename T0, typename T1, typename T2,
        class Sat = xetla_saturation_off_tag>
__XETLA_API typename std::remove_const<T0>::type xetla_lsr(
        T1 src0, T2 src1, Sat sat = {}) {
    return cm_lsr<T0, T1, T2>(src0, src1, Sat::value);
}

/// Arithmetical Shift Right (vector version)
/// @tparam T0 element type of the returned vector. Must be any integer type.
/// @tparam T1 element type of the input vector. Must be any integer type.
/// @tparam SZ size of the input and returned vectors.
/// @tparam U type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input vector.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of shifted elements.
template <typename T0, typename T1, int SZ, typename U,
        class Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T0, SZ> xetla_asr(
        xetla_vector<T1, SZ> src0, U src1, Sat sat = {}) {
    return cm_asr<T0, T1, SZ, U>(src0, src1, Sat::value);
}

/// Arithmetical Shift Right (scalar version)
/// @tparam T0 element type of the returned value. Must be any integer type.
/// @tparam T1 element type of the input value \p src0. Must be any integer
/// type.
/// @tparam T2 type of scalar operand \p src1. Must be any integer type.
/// @param src0 the input value.
/// @param src1 the number of bit positions the input vector shall be shifted.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return shifted value.
template <typename T0, typename T1, typename T2,
        class Sat = xetla_saturation_off_tag>
__XETLA_API typename std::remove_const<T0>::type xetla_asr(
        T1 src0, T2 src1, Sat sat = {}) {
    return cm_asr<T0, T1, T2>(src0, src1, Sat::value);
}

/// Pack a xetla_mask into a single unsigned 32-bit integer value.
/// i'th bit in the returned value is set to the result of comparison of the
/// i'th element of the input argument to zero. "equals to zero" gives \c 0,
/// "not equal to zero" gives \c 1. Remaining (if any) bits if the result are
/// filled with \c 0.
/// @tparam N Size of the input mask.
/// @param src0 The input mask.
/// @return The packed mask as an <code> unsigned int</code> 32-bit value.
template <int N>
__XETLA_API uint32_t xetla_pack_mask(xetla_mask<N> src0) {
    return cm_pack_mask<N>(src0);
}

/// Unpack an unsigned 32-bit integer value into a xetla_mask. Only \c N least
/// significant bits are used, where \c N is the number of elements in the
/// result mask. Each input bit is stored into the corresponding vector element
/// of the output mask.
/// @tparam N Size of the output mask.
/// @param src0 The input packed mask.
/// @return The unpacked mask as a xetla_mask object.
template <int N>
__XETLA_API xetla_mask<N> xetla_unpack_mask(uint32_t src0) {
    return cm_unpack_mask<uint16_t, N>(src0);
}

/// @} xetla_core_bit_manipulation

} // namespace gpu::xetla
