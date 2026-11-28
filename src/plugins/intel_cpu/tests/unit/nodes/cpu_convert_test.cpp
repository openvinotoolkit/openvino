// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/common/cpu_convert.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

using namespace ov::intel_cpu;

namespace {

template <typename T>
std::vector<uint8_t> convert_to_boolean(const std::vector<T>& src, ov::element::Type src_prc, bool parallel) {
    std::vector<uint8_t> dst(src.size(), 0xFF);
    if (parallel) {
        cpu_parallel_convert(src.data(), dst.data(), src_prc, ov::element::boolean, src.size());
    } else {
        cpu_convert(src.data(), dst.data(), src_prc, ov::element::boolean, src.size());
    }
    return dst;
}

template <typename T>
void check_convert_to_boolean(ov::element::Type src_prc) {
    const std::vector<T> src = {T(0), T(0.25), T(-0.25), T(1), T(-1), T(2), T(300), T(-300)};
    const std::vector<uint8_t> expected = {0, 1, 1, 1, 1, 1, 1, 1};
    EXPECT_EQ(convert_to_boolean(src, src_prc, false), expected);
    EXPECT_EQ(convert_to_boolean(src, src_prc, true), expected);
}

template <typename T>
void check_integral_convert_to_boolean(ov::element::Type src_prc) {
    std::vector<T> src = {T(0), T(1), T(2), std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
    std::vector<uint8_t> expected = {0, 1, 1, 1, std::numeric_limits<T>::lowest() != 0};
    if (std::is_signed_v<T>) {
        src.push_back(T(-1));
        expected.push_back(1);
    }
    EXPECT_EQ(convert_to_boolean(src, src_prc, false), expected);
    EXPECT_EQ(convert_to_boolean(src, src_prc, true), expected);
}

}  // namespace

TEST(CpuConvertTest, FloatingPointToBoolean) {
    check_convert_to_boolean<float>(ov::element::f32);
    check_convert_to_boolean<double>(ov::element::f64);
    check_convert_to_boolean<ov::float16>(ov::element::f16);
    check_convert_to_boolean<ov::bfloat16>(ov::element::bf16);
}

TEST(CpuConvertTest, FloatingPointSpecialValuesToBoolean) {
    const std::vector<float> src = {-0.0F,
                                    std::numeric_limits<float>::denorm_min(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::quiet_NaN()};
    const std::vector<uint8_t> expected = {0, 1, 1, 1, 1};
    EXPECT_EQ(convert_to_boolean(src, ov::element::f32, false), expected);
}

TEST(CpuConvertTest, IntegralToBoolean) {
    check_integral_convert_to_boolean<uint8_t>(ov::element::u8);
    check_integral_convert_to_boolean<int8_t>(ov::element::i8);
    check_integral_convert_to_boolean<uint16_t>(ov::element::u16);
    check_integral_convert_to_boolean<int16_t>(ov::element::i16);
    check_integral_convert_to_boolean<uint32_t>(ov::element::u32);
    check_integral_convert_to_boolean<int32_t>(ov::element::i32);
    check_integral_convert_to_boolean<uint64_t>(ov::element::u64);
    check_integral_convert_to_boolean<int64_t>(ov::element::i64);
}

TEST(CpuConvertTest, BooleanToBooleanIsCopied) {
    const std::vector<uint8_t> src = {0, 1, 1, 0};
    EXPECT_EQ(convert_to_boolean(src, ov::element::boolean, false), src);
}
