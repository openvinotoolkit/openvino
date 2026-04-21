// Copyright (C) 2018-2026 Intel Corporation
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

/// Maximum tolerated |abs(src - round_trip_f16(src))| for an in-range element.
/// If any element exceeds this threshold, the whole tensor is kept in FP32
/// (acts as a tensor-wide veto in check_f16_compression()).
inline constexpr double f16_compression_max_abs_error = 1.0;
/// Maximum tolerated relative round-trip error |abs_diff / src| for an
/// in-range element. Elements above this threshold are accumulated into the
/// combined rejection count used together with f16_compression_keep_threshold.
inline constexpr double f16_compression_max_rel_error = 1e-4;
/// Proportion of combined rejections (out-of-FP16-range + high relative-error)
/// at which the Constant is kept in FP32 (no FP16 compression).
inline constexpr float f16_compression_keep_threshold = 0.75f;

// Single-pass combined check for FP16 compression feasibility.
// Bails immediately if any in-range value has significant precision loss
// (|round-trip error| > f16_compression_max_abs_error). Otherwise accumulates a
// combined rejection count used by CompressFloatConstantsImpl to decide whether
// to keep the Constant in FP32 via the f16_compression_keep_threshold.
// JIT/AVX2+F16C accelerated on x86.
struct CompressionCheckResult {
    // Combined count of rejected elements: values outside finite FP16 range PLUS
    // in-range values whose FP16 conversion exceeds f16_compression_max_rel_error.
    // This is NOT a pure out-of-range count — see count_out_of_f16_range() for
    // that.
    size_t out_of_range_count;
    // Early-bail flag: true iff any in-range element has |abs error| greater
    // than f16_compression_max_abs_error after the FP16 round-trip. When set,
    // out_of_range_count is not guaranteed to be complete.
    bool has_lossy;
};
CompressionCheckResult check_f16_compression(const float* arg, size_t count);

// Counts elements in `arg` that fall outside the finite FP16 range (subnormal
// when rounded, or larger in magnitude than float16::max()). Distinct from
// check_f16_compression(), which also accounts for relative-error rejections.
size_t count_out_of_f16_range(const float* arg, size_t count);

// Convert values from f32 to f16 with clamping to f16 min/max when value is out of normal finite numbers range
void convert_from_f32_to_f16_with_clamp(const float* arg, float16* out, size_t count);

// Convert values from bf16 to f16 with clamping to f16 min/max when value is out of normal finite numbers range
void convert_from_bf16_to_f16_with_clamp(const bfloat16* arg, float16* out, size_t count);
}  // namespace reference
}  // namespace ov
