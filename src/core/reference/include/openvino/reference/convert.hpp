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

constexpr double f16_compression_max_abs_error = 1.0;
// FP16 has a 10-bit mantissa, so the natural per-element relative round-trip
// error is ~2^-11 (≈4.88e-4). This threshold is intentionally tighter than
// that — it's meaningful only for scalar (numel==1) constants used as
// mathematical scale factors (e.g. log(16) in attention bucketing), where a
// rounded value cascades through every dependent computation. Tensor-wide
// compression decisions rely on the abs-error veto (has_lossy) instead.
constexpr double f16_compression_max_rel_error = 1e-4;
constexpr float f16_compression_keep_threshold = 0.75f;

// Single-pass check for FP16 compression feasibility. Bails immediately if any
// in-range value has significant precision loss (|abs error| >
// f16_compression_max_abs_error). Otherwise counts values outside the finite
// FP16 range; CompressFloatConstantsImpl uses that count with
// f16_compression_keep_threshold to decide whether to keep the Constant in
// FP32. JIT/AVX2+F16C accelerated on x86.
struct CompressionCheckResult {
    // Number of elements whose FP16 representation falls outside the finite
    // FP16 range: subnormal when rounded (|v| < float16::from_bits(0x0001))
    // or larger in magnitude than float16::max(). Same semantics as
    // count_out_of_f16_range().
    size_t out_of_range_count;
    // Early-bail flag: true iff any in-range element has |abs error| greater
    // than f16_compression_max_abs_error after the FP16 round-trip. When set,
    // out_of_range_count is not guaranteed to be complete.
    bool has_lossy;
};
CompressionCheckResult check_f16_compression(const float* arg, size_t count);

// Counts elements in `arg` that fall outside the finite FP16 range (subnormal
// when rounded, or larger in magnitude than float16::max()). Back-compat shim
// for external developer-package consumers that linked against the pre-JIT
// symbol; the in-tree compression path uses check_f16_compression().
size_t count_out_of_f16_range(const float* arg, size_t count);

// Convert values from f32 to f16 with clamping to f16 min/max when value is out of normal finite numbers range
void convert_from_f32_to_f16_with_clamp(const float* arg, float16* out, size_t count);

// Convert values from bf16 to f16 with clamping to f16 min/max when value is out of normal finite numbers range
void convert_from_bf16_to_f16_with_clamp(const bfloat16* arg, float16* out, size_t count);
}  // namespace reference
}  // namespace ov
