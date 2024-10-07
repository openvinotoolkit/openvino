// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/type/nf4.hpp"

#if !defined(OS_CHROMEOS) && (defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64))
#    define OV_CORE_USE_XBYAK_JIT
#endif

#if defined(OS_CHROMEOS) && defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX2)
#    define OV_CORE_USE_INTRINSICS
#endif

namespace ov {

template <class ElementIter>
constexpr bool is_nf4_iterator() {
    using it = typename std::decay<ElementIter>::type;
    using T = fundamental_type_for<element::nf4>;
    return std::is_same<it, element::Iterator<element::nf4, const T>>::value ||
           std::is_same<it, element::Iterator<element::nf4, T>>::value;
}

namespace reference {
namespace detail {

template <typename TI, typename TO>
constexpr typename std::enable_if<!std::is_same<TO, char>::value, TO>::type convert(const TI v) {
    return static_cast<TO>(v);
}

template <typename TI, typename TO>
constexpr typename std::enable_if<std::is_same<TO, char>::value, TO>::type convert(const TI v) {
    return static_cast<char>(static_cast<bool>(v));
}
}  // namespace detail

template <typename InputIt, typename OutputIt>
void convert(InputIt arg, OutputIt out, const size_t count) {
    using IN_T = typename std::iterator_traits<InputIt>::value_type;
    using OUT_T = typename std::iterator_traits<OutputIt>::value_type;

    // Deduce types for NF4 <-> floating point conversion to use quantization.
    using From = typename std::
        conditional<is_nf4_iterator<InputIt>() && !std::is_integral<OUT_T>::value, const float, IN_T>::type;
    using To =
        typename std::conditional<is_nf4_iterator<OutputIt>() && !std::is_integral<IN_T>::value, float, OUT_T>::type;

    std::transform(arg, arg + count, out, detail::convert<From, To>);
}

template <typename TI, typename TO>
void convert(const TI* arg, TO* out, const size_t count) {
    std::transform(arg, arg + count, out, detail::convert<TI, TO>);
}

template <>
void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count);
template <>
void convert<float16, float>(const float16* arg, float* out, size_t count);
template <>
void convert<float, float16>(const float* arg, float16* out, size_t count);
template <>
void convert<float, int8_t>(const float* arg, int8_t* out, size_t count);
template <>
void convert<float16, int8_t>(const float16* arg, int8_t* out, size_t count);
template <>
void convert<bfloat16, float16>(const bfloat16* arg, float16* out, size_t count);
template <>
void convert<bfloat16, float>(const bfloat16* arg, float* out, size_t count);

template <>
void convert<int32_t, float16>(const int32_t* arg, float16* out, size_t count);

// Count how many f32 values is out of normal finite numbers range when converted to f16
size_t count_out_of_f16_range(const float* arg, size_t count);

// Convert values from f32 to f16 with clamping to f16 min/max when value is out of normal finite numbers range
void convert_from_f32_to_f16_with_clamp(const float* arg, float16* out, size_t count);

// Convert values from bf16 to f16 with clamping to f16 min/max when value is out of normal finite numbers range
void convert_from_bf16_to_f16_with_clamp(const bfloat16* arg, float16* out, size_t count);
}  // namespace reference
}  // namespace ov
