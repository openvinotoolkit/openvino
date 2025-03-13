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
#else
#include "common/core/cm/base_ops.hpp"
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_core_math
/// @{

/// Get absolute value (vector version)
/// @tparam T0 element type of the returned vector.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input and returned vector.
/// @param src0 the input vector.
/// @return vector of absolute values.
template <typename T0, typename T1, int SZ>
__XETLA_API xetla_vector<T0, SZ> xetla_abs(xetla_vector<T1, SZ> src0) {
    static_assert(!(is_internal_type<T0>::value || is_internal_type<T1>::value),
            "The internal types are not yet supported!");
    return cm_abs<T0, T1, SZ>(src0);
}

/// Get absolute value (scalar version)
/// @tparam T0 element type of the returned value.
/// @tparam T1 element type of the input value.
/// @param src0 the source operand.
/// @return absolute value.
template <typename T0, typename T1>
std::enable_if_t<!std::is_same<remove_const_t<T0>, remove_const_t<T1>>::value,
        remove_const_t<T0>>
        __XETLA_API xetla_abs(T1 src0) {
    static_assert(!(is_internal_type<T0>::value || is_internal_type<T1>::value),
            "The internal types are not yet supported!");
    return cm_abs<T0, T1>(src0);
}

/// Get absolute value (vector version). This is a specialization of a version
/// with three template parameters, where the element types of the input and
/// output vector are the same.
/// @tparam T1 element type of the input and output vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @return vector of absolute values.
template <typename T1, int SZ>
__XETLA_API xetla_vector<T1, SZ> xetla_abs(xetla_vector<T1, SZ> src0) {
    static_assert(!(is_internal_type<T1>::value),
            "The internal types are not yet supported!");
    return cm_abs<T1, SZ>(src0);
}

/// Get absolute value (scalar version). This is a specialization of a version
/// with two template parameters, where the types of the input and output value
/// are the same.
/// @tparam T1 element type of the input and output value.
/// @param src0 the source operand.
/// @return absolute value.
template <typename T1>
__XETLA_API typename std::remove_const<T1>::type xetla_abs(T1 src0) {
    static_assert(!(is_internal_type<T1>::value),
            "The internal types are not yet supported!");
    return cm_abs<T1>(src0);
}

/// Selects component-wise the maximum of the two vectors.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.

template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_max(
        xetla_vector<T, SZ> src0, xetla_vector<T, SZ> src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_max<T, T, SZ, xetla_vector<T, SZ>>(src0, src1, Sat::value);
}

/// Selects maximums for each element of the input vector and a scalar.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_max(
        xetla_vector<T, SZ> src0, T src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_max<T, T, SZ, T>(src0, src1, Sat::value);
}

/// Selects maximums for each element of the input scalar and a vector.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the scalar value.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise maximum elements.
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_max(
        T src0, xetla_vector<T, SZ> src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_max<T, T, xetla_vector<T, SZ>>(src0, src1, Sat::value);
}

/// Selects maximum between two scalar values.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @param src0 the scalar value.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return maximum value between the two inputs.
template <typename T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_max(T src0, T src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_max<T, T, T>(src0, src1, Sat::value);
}

/// Selects component-wise the minimum of the two vectors.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.

template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_min(
        xetla_vector<T, SZ> src0, xetla_vector<T, SZ> src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_min<T, T, SZ, xetla_vector<T, SZ>>(src0, src1, Sat::value);
}

/// Selects minimums for each element of the input vector and a scalar.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_min(
        xetla_vector<T, SZ> src0, T src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_min<T, T, SZ, T>(src0, src1, Sat::value);
}

/// Selects minimums for each element of the input scalar and a vector.
/// The source operands must be both of integer or both of
/// floating-point type.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the scalar value.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise minimum elements.
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_min(
        T src0, xetla_vector<T, SZ> src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_min<T, T, xetla_vector<T, SZ>>(src0, src1, Sat::value);
}

/// Selects minimum between two scalar values.
/// The source operands must be both of integer or both of floating-point type.
/// @tparam T element type of the input and return vectors.
/// @param src0 the scalar value.
/// @param src1 the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return minimum value between the two inputs.
template <typename T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_min(T src0, T src1, Sat sat = {}) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_min<T, T, T>(src0, src1, Sat::value);
}

/// @brief Calculate exponent value for each element of the input vector, the base is e.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise exponent elements.
template <class T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_exp(
        xetla_vector<T, SZ> src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    constexpr float log2e = 1.442695f;
    return cm_exp<SZ>(xetla_vector<T, SZ>(src * log2e), Sat::value);
}

/// @brief Calculate exponent value of a scalar, the base is e.
/// @tparam T element type of the input and return a scalar.
/// @param src the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return exponent value.
template <class T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_exp(T src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    constexpr float log2e = 1.442695f;
    return cm_exp(T(src * log2e), Sat::value);
}

/// @brief Calculate exponent value for each element of the input vector, the base is 2.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise exponent elements.
template <class T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_exp2(
        xetla_vector<T, SZ> src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_exp<SZ>(src, Sat::value);
}

/// @brief Calculate exponent value of a scalar, the base is 2.
/// @tparam T element type of the input and return a scalar.
/// @param src the scalar value.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return exponent value.
template <class T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_exp2(T src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_exp(src, Sat::value);
}

/// @brief Calculate the inversion, i.e. 1/x (vector version).
/// @tparam T Is the element data type
/// @tparam SZ Is the element num in the vector
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of 1/x
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_inv(
        xetla_vector<T, SZ> src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_inv(src, Sat::value);
}

/// @brief Calculate the inversion, i.e. 1/x (scalar version).
/// @tparam T Is the element data type
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input scalar.
/// @return Returns the result of 1/x
template <typename T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_inv(T src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_inv(src, Sat::value);
}

/// @brief Calculate the square root, i.e. x^(1/2), this is not IEEE754-compatible (vector version).
/// @tparam T Is the element data type
/// @tparam SZ Is the element num in the vector
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of x^(1/2)
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_sqrt(
        xetla_vector<T, SZ> src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_sqrt(src, Sat::value);
}

/// @brief Calculate the square root, i.e. x^(1/2), this is not IEEE754-compatible (scalar version).
/// @tparam T Is the element data type
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of x^(1/2)
template <typename T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_sqrt(T src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_sqrt(src, Sat::value);
}

/// @brief Calculate the square root, i.e. x^(1/2), IEEE754-compatible (vector version).
/// @tparam T Is the element data type
/// @tparam SZ Is the element num in the vector
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of x^(1/2)
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_sqrt_ieee(
        xetla_vector<T, SZ> src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, double>::value),
            "Only support fp32 and fp16");
    return cm_sqrt_ieee(src, Sat::value);
}

/// @brief Calculate the square root, i.e. x^(1/2), IEEE754-compatible (scalar version).
/// @tparam T Is the element data type
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of x^(1/2)
template <typename T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_sqrt_ieee(T src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, double>::value),
            "Only support fp32 and fp16");
    return cm_sqrt_ieee(src, Sat::value);
}

/// @brief Calculate the inversion of square root, i.e. 1/sqrt(x) (vector version).
/// @tparam T Is the element data type
/// @tparam SZ Is the element num in the vector
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of 1/sqrt(x)
template <typename T, int SZ, typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T, SZ> xetla_rsqrt(
        xetla_vector<T, SZ> src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_rsqrt(src, Sat::value);
}

/// @brief Calculate the inversion of square root, i.e. 1/sqrt(x) (scalar version).
/// @tparam T Is the element data type
/// @tparam Sat Is the saturation flag(off by default). Possible values: saturation_on/saturation_off.
/// @param src Is the input vector.
/// @return Returns the result of 1/sqrt(x)
template <typename T, typename Sat = xetla_saturation_off_tag>
__XETLA_API T xetla_rsqrt(T src, Sat sat = {}) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    return cm_rsqrt(src, Sat::value);
}

/// @brief Calculate the tanh (vector version).
/// @tparam T Is the element data type
/// @tparam SZ Is the element num in the vector
/// @param src Is the input vector.
/// @return Returns the result of tanh(x)
template <typename T, int SZ>
__XETLA_API xetla_vector<T, SZ> xetla_tanh(xetla_vector<T, SZ> src) {
    static_assert(std::is_same<remove_const_t<T>, float>::value,
            "Only support fp32! ");
    constexpr uint32_t flag_elems = 8 * 16;
    xetla_vector<T, SZ> ret;
#pragma unroll
    for (int i = 0; i < SZ / flag_elems; i++) {
        auto src_sub = src.xetla_select<flag_elems, 1>(i * flag_elems);
        auto ret_sub = ret.xetla_select<flag_elems, 1>(i * flag_elems);
        xetla_mask<flag_elems> mask = src_sub >= 10;
        xetla_vector<T, flag_elems> exp2x
                = xetla_exp<T, flag_elems>(src_sub * 2.f);
        ret_sub = (exp2x - 1.f) / (exp2x + 1.f);
        xetla_vector<T, flag_elems> ones(1);
        ret_sub.xetla_merge(ones, mask);
    }
    if constexpr (SZ % flag_elems != 0) {
        constexpr uint32_t start_pos = SZ / flag_elems * flag_elems;
        constexpr uint32_t remain_elems = SZ % flag_elems;

        auto src_sub = src.xetla_select<remain_elems, 1>(start_pos);
        auto ret_sub = ret.xetla_select<remain_elems, 1>(start_pos);
        xetla_mask<remain_elems> mask = src_sub >= 10;
        xetla_vector<T, remain_elems> exp2x
                = xetla_exp<T, remain_elems>(src_sub * 2.f);
        ret_sub = (exp2x - 1.f) / (exp2x + 1.f);
        xetla_vector<T, remain_elems> ones(1);
        ret_sub.xetla_merge(ones, mask);
    }

    return ret;
}

/// @brief Calculate the tanh (scalar version).
/// @tparam T Is the element data type
/// @tparam SZ Is the element num in the vector
/// @param src Is the input vector.
/// @return Returns the result of tanh(x)
template <typename T>
__XETLA_API T xetla_tanh(T src) {
    static_assert(std::is_same<remove_const_t<T>, float>::value,
            "Only support fp32! ");
    T exp2x = xetla_exp<T>(src * 2.f);
    T ret = (exp2x - 1.f) / (exp2x + 1.f);
    return (src >= 10) ? 1 : ret;
}

/// @brief Calculate sigmoid value for each element of the input vector.
/// @tparam T element type of the input and return vectors.
/// @tparam SZ size of the input and returned vectors.
/// @param src the input vector.
/// @return vector of sigmoid of component-wise elements.
template <typename T, int SZ>
__XETLA_API xetla_vector<T, SZ> xetla_sigmoid(xetla_vector<T, SZ> src) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    xetla_mask<SZ> mask = src <= -10;
    xetla_vector<T, SZ> exp = xetla_exp<T, SZ>(-src);
    xetla_vector<T, SZ> ret_sub = 1.f / (exp + 1.f);
    ret_sub.xetla_merge(0, mask);

    return ret_sub;
}

/// @brief Calculate sigmoid of a scalar.
/// @tparam T element type of the input and return a scalar.
/// @param src the scalar value.
/// @return sigmoid value.
template <typename T>
__XETLA_API T xetla_sigmoid(T src) {
    static_assert((std::is_same<remove_const_t<T>, float>::value)
                    || (std::is_same<remove_const_t<T>, fp16>::value),
            "Only support fp32 and fp16");
    T exp = xetla_exp<T>(-src);
    T ret = 1.f / (exp + 1.f);
    return (src <= -10) ? 0 : ret;
}

/// Add two unsigned integer vectors, return the result and in-place update the carry.
/// @tparam T element type of the src, should be uint32_t.
/// @tparam SZ element num of the vector.
/// @param src0 is the src0 of the add operation.
/// @param src1 is the src1 of the add operation.
/// @param carry is the carry of the add operation.
/// @return result of the src0 + src1.
template <typename T, int SZ>
__XETLA_API xetla_vector<T, SZ> xetla_add_c(xetla_vector<T, SZ> src0,
        xetla_vector<T, SZ> src1, xetla_vector_ref<T, SZ> __REF__ carry) {
    static_assert((std::is_same<remove_const_t<T>, uint32_t>::value),
            "For addc, only uint32_t is supported");
    return cm_addc(src0, src1, carry);
}

/// Add one unsigned integer vectors with a scalar, return the result and in-place update the carry.
/// @tparam T element type of the src, should be uint32_t.
/// @tparam SZ element num of the vector.
/// @param src0 is the src0 of the add operation.
/// @param src1 is the src1 of the add operation.
/// @param carry is the carry of the add operation.
/// @return result of the src0 + src1.
template <typename T, int SZ>
__XETLA_API xetla_vector<T, SZ> xetla_add_c(xetla_vector<T, SZ> src0, T src1,
        xetla_vector_ref<T, SZ> __REF__ carry) {
    static_assert((std::is_same<remove_const_t<T>, uint32_t>::value),
            "For addc, only uint32_t is supported");
    return cm_addc(src0, src1, carry);
}

/// @brief Multiply src0 with src1, return the hi part and in-place update the lo part.
/// @tparam T0 Return data type, should be 32 bit integer.
/// @tparam T1 Type of src0, should be 32 bit integer.
/// @tparam T2 Type of src1, should be 32 bit integer.
/// @tparam SZ Element num of the vector.
/// @param lo is the low 32 bit part of the result.
/// @param src0 is the src0 of the mul operation.
/// @param src1 is the src1 of the mul operation.
/// @return return the high 32 bit part of the result.
template <typename T0, typename T1, typename T2, int SZ>
__XETLA_API xetla_vector<T0, SZ> xetla_imul(xetla_vector_ref<T0, SZ> __REF__ lo,
        xetla_vector<T1, SZ> src0, T2 src1) {
    return cm_imul(lo, src0, src1);
}

/// Performs reduction over elements of the input vector.
/// @tparam T0 type of the return value.
/// @tparam T1 element type of the input vector.
/// @tparam SZ size of the input vector.
/// @tparam BinaryOperation type representing the operation. Can be an
///   instantion of one of the following types:
///   \li \c reduce_op::sum, performs reduce sum operation
///   \li \c reduce_op::prod, performs reduce prod operation
///   \li \c reduce_op::min, performs reduce min operation
///   \li \c reduce_op::max, performs reduce max operation
/// @param v the vector to perform reduction on
/// @return result of the reduction
template <typename T0, typename T1, int SZ, reduce_op BinaryOperation>
__XETLA_API T0 xetla_reduce(xetla_vector<T1, SZ> v) {
    if constexpr (BinaryOperation == reduce_op::sum) {
        return cm_sum<T0, T1, SZ>(v);
    } else if constexpr (BinaryOperation == reduce_op::prod) {
        return cm_prod<T0, T1, SZ>(v);
    } else if constexpr (BinaryOperation == reduce_op::min) {
        return cm_reduced_min<T0, T1, SZ>(v);
    } else if constexpr (BinaryOperation == reduce_op::max) {
        return cm_reduced_max<T0, T1, SZ>(v);
    }
}

/// Get rounded value
/// @tparam T element type of the input vector.
/// @tparam SZ size of the input and returned vector.
/// @param src0 the input vector.
/// @return vector of rounded values.
template <typename T, int SZ>
__XETLA_API xetla_vector<T, SZ> xetla_rnde(xetla_vector<T, SZ> src0) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return cm_rnde<T, SZ>(src0);
}

/// Adds two vectors with saturation
/// The source operands must be  both of floating-point type.
/// @tparam T0 element type of the input vectors.
/// @tparam T1 element type of the return vector.
/// @tparam SZ size of the input and returned vectors.
/// @param src0 the input vector.
/// @param src1 the input vector.
/// @param sat enables/disables the saturation (off by default). Possible
/// values: saturation_on/saturation_off.
/// @return vector of component-wise additive elements.
template <typename T1, typename T0, int SZ,
        typename Sat = xetla_saturation_off_tag>
__XETLA_API xetla_vector<T1, SZ> xetla_add(
        xetla_vector<T0, SZ> src0, xetla_vector<T0, SZ> src1, Sat sat = {}) {
    static_assert(
            !((is_internal_type<T0>::value) || (is_internal_type<T1>::value)),
            "The internal types are not yet supported!");
    return cm_add<T1>(src0, src1, sat.value);
}

/// Saturation function.
/// @tparam T0 element type of the input vectors.
/// @tparam T1 element type of the return vector.
/// @tparam SZ size of the input and returned vectors.
template <typename T1, typename T0, int SZ>
__XETLA_API xetla_vector<T1, SZ> xetla_sat(xetla_vector<T0, SZ> src) {
    static_assert(
            !((is_internal_type<T0>::value) || (is_internal_type<T1>::value)),
            "The internal types are not yet supported!");
    return xetla_vector<T1, SZ>(src, SAT);
}

/// Count number of bits set in the source operand per element.
/// @param src0 the source operand to count bits in.
/// @return a vector of \c uint32_t, where each element is set to bit count of
///     the corresponding element of the source operand.
template <typename T, int N>
__XETLA_API std::enable_if_t<std::is_integral<T>::value && (sizeof(T) <= 4),
        xetla_vector<uint32_t, N>>
xetla_cbit(xetla_vector<T, N> src) {
    return cm_cbit<T, N>(src);
}

/// Scalar version of \c cbit - both input and output are scalars rather
/// than vectors.
template <typename T>
__XETLA_API std::enable_if_t<std::is_integral<T>::value && (sizeof(T) <= 4),
        uint32_t>
xetla_cbit(T src) {
    return cm_cbit<T>(src);
}

/// @} xetla_core_math

} // namespace gpu::xetla
