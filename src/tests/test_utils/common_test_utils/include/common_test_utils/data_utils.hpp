// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/tensor.hpp"

namespace NGraphFunctions {
namespace Utils {

template <ov::element::Type_t dType>
std::vector<typename ov::element_type_traits<dType>::value_type> inline generateVector(
    size_t vec_len,
    typename ov::element_type_traits<dType>::value_type upTo = 10,
    typename ov::element_type_traits<dType>::value_type startFrom = 1,
    int32_t seed = 1) {
    using dataType = typename ov::element_type_traits<dType>::value_type;
    std::vector<dataType> res(vec_len);

    std::mt19937 gen(seed);
    if (std::is_floating_point<dataType>()) {
        // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
        std::uniform_real_distribution<double> dist(static_cast<double>(startFrom), static_cast<double>(upTo));
        // explicitly include data range borders to avoid missing the corner values while data generation
        res[0] = startFrom;
        res[vec_len - 1] = upTo;
        for (size_t i = 1; i < vec_len - 1; i++) {
            res[i] = static_cast<dataType>(dist(gen));
        }
        return res;
    } else if (std::is_same<bool, dataType>()) {
        std::bernoulli_distribution dist;
        for (size_t i = 0; i < vec_len; i++) {
            res[i] = static_cast<dataType>(dist(gen));
        }
        return res;
    } else {
        // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
        std::uniform_int_distribution<long> dist(static_cast<long>(startFrom), static_cast<long>(upTo));
        // explicitly include data range borders to avoid missing the corner values while data generation
        res[0] = startFrom;
        res[vec_len - 1] = upTo;
        for (size_t i = 1; i < vec_len - 1; i++) {
            res[i] = static_cast<dataType>(dist(gen));
        }
        return res;
    }
}

template <>
std::vector<ov::float16> inline generateVector<ov::element::Type_t::f16>(size_t vec_len,
                                                                         ov::float16 upTo,
                                                                         ov::float16 startFrom,
                                                                         int32_t seed) {
    std::vector<ov::float16> res(vec_len);
    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_real_distribution<float> dist(startFrom, upTo);
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = ov::float16(dist(gen));
    }
    return res;
}

template <>
std::vector<ov::bfloat16> inline generateVector<ov::element::Type_t::bf16>(size_t vec_len,
                                                                           ov::bfloat16 upTo,
                                                                           ov::bfloat16 startFrom,
                                                                           int32_t seed) {
    std::vector<ov::bfloat16> res(vec_len);

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_real_distribution<float> dist(startFrom, upTo);
    ;
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = ov::bfloat16(dist(gen));
    }
    return res;
}

template <typename fromType, typename toType>
std::vector<toType> castVector(const std::vector<fromType>& vec) {
    std::vector<toType> resVec;
    resVec.reserve(vec.size());
    for (const auto& el : vec) {
        resVec.push_back(static_cast<toType>(el));
    }
    return resVec;
}

}  // namespace Utils
}  // namespace NGraphFunctions

namespace ov {
namespace test {
namespace utils {

inline void fill_data(float* data, size_t size, size_t duty_ratio = 10) {
    for (size_t i = 0; i < size; i++) {
        if ((i / duty_ratio) % 2 == 1) {
            data[i] = 0.0f;
        } else {
            data[i] = sin(static_cast<float>(i));
        }
    }
}

/**
 * @brief Create vector of floats with length of vec_len, with values ranging from min to max,
 * with initial seed equal to variable seed with default of 0
 */
inline std::vector<float> generate_float_numbers(std::size_t vec_len, float min, float max, int seed = 0) {
    std::vector<float> res;
    std::mt19937 gen(seed);

    std::uniform_real_distribution<float> dist(min, max);
    for (std::size_t i = 0; i < vec_len; i++)
        res.emplace_back(static_cast<float>(dist(gen)));

    return res;
}

/**
 * Fill tensor with value data. Broadcast semantic is included.
 * Broadcasting with alignment through last dimension.
 *
 * @param tensor tensor to fill in
 * @param values src tensor which should be broadcast
 */
void fill_data_with_broadcast(ov::Tensor& tensor, ov::Tensor& values);

/**
 * Wrapper on top of fill_data_with_broadcast with simplified signature
 *
 * @param tensor tensor to fill in
 * @param axis Axis to apply values
 * @param values data to broadcast
 */
void fill_data_with_broadcast(ov::Tensor& tensor, size_t axis, std::vector<float> values);

ov::Tensor make_tensor_with_precision_convert(const ov::Tensor& tensor, ov::element::Type prc);

template <typename T>
inline void fill_roi_raw_ptr(T* data,
                             size_t data_size,
                             const uint32_t range,
                             const int32_t height,
                             const int32_t width,
                             const float omega,
                             const bool is_roi_max_mode,
                             const int32_t seed = 1) {
    std::default_random_engine random(seed);
    std::uniform_int_distribution<int32_t> distribution(0, range);

    const int max_y = (is_roi_max_mode) ? (height - 1) : 1;
    const int max_x = (is_roi_max_mode) ? (width - 1) : 1;

    float center_h = (max_y) / 2.0f;
    float center_w = (max_x) / 2.0f;

    for (size_t i = 0; i < data_size; i += 5) {
        data[i] = static_cast<T>(distribution(random));
        const float x0 = (center_w + width * 0.3f * sin(static_cast<float>(i + 1) * omega));
        const float x1 = (center_w + width * 0.3f * sin(static_cast<float>(i + 3) * omega));
        data[i + 1] = static_cast<T>(is_roi_max_mode ? std::floor(x0) : x0);
        data[i + 3] = static_cast<T>(is_roi_max_mode ? std::floor(x1) : x1);
        if (data[i + 3] < data[i + 1]) {
            std::swap(data[i + 1], data[i + 3]);
        }
        if (data[i + 1] < 0)
            data[i + 1] = 0;
        if (data[i + 3] > max_x)
            data[i + 3] = static_cast<T>(max_x);

        const float y0 = (center_h + height * 0.3f * sin(static_cast<float>(i + 2) * omega));
        const float y1 = (center_h + height * 0.3f * sin(static_cast<float>(i + 4) * omega));
        data[i + 2] = static_cast<T>(is_roi_max_mode ? std::floor(y0) : y0);
        data[i + 4] = static_cast<T>(is_roi_max_mode ? std::floor(y1) : y1);
        if (data[i + 4] < data[i + 2]) {
            std::swap(data[i + 2], data[i + 4]);
        }
        if (data[i + 2] < 0)
            data[i + 2] = 0;
        if (data[i + 4] > max_y)
            data[i + 4] = static_cast<T>(max_y);
    }
}

void fill_psroi(ov::Tensor& tensor,
                int batchSize,
                int height,
                int width,
                int groupSize,
                float spatialScale,
                int spatialBinsX,
                int spatialBinsY,
                const std::string& mode);

void fill_data_roi(ov::Tensor& tensor,
                   const uint32_t range,
                   const int height,
                   const int width,
                   const float omega,
                   const bool is_roi_max_mode,
                   const int seed = 1);

template <class T>
void inline fill_data_random(T* pointer,
                             std::size_t size,
                             const uint32_t range = 10,
                             double_t start_from = 0,
                             const int32_t k = 1,
                             const int seed = 1) {
    if (range == 0) {
        for (std::size_t i = 0; i < size; i++) {
            pointer[i] = static_cast<T>(start_from);
        }
        return;
    }

    testing::internal::Random random(seed);
    const uint32_t k_range = k * range;  // range with respect to k
    random.Generate(k_range);

    if (start_from < 0 && !std::numeric_limits<T>::is_signed) {
        start_from = 0;
    }
    for (std::size_t i = 0; i < size; i++) {
        pointer[i] = static_cast<T>(start_from + static_cast<T>(random.Generate(k_range)) / k);
    }
}

template <class T>
void inline fill_data_ptr_real_random_float(T* pointer,
                                            std::size_t size,
                                            const float min,
                                            const float max,
                                            const int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min, max);

    for (std::size_t i = 0; i < size; i++) {
        pointer[i] = static_cast<T>(dist(gen));
    }
}

template <class T>
void inline fill_data_random_act_dft(T* pointer,
                                     std::size_t size,
                                     const uint32_t range = 10,
                                     double_t start_from = 0,
                                     const int32_t k = 1,
                                     const int seed = 1) {
    if (range == 0) {
        for (std::size_t i = 0; i < size; i++) {
            pointer[i] = static_cast<T>(start_from);
        }
        return;
    }

    testing::internal::Random random(seed);
    const uint32_t k_range = k * range;  // range with respect to k
    random.Generate(k_range);

    if (start_from < 0 && !std::numeric_limits<T>::is_signed) {
        start_from = 0;
    }
    for (std::size_t i = 0; i < size; i++) {
        pointer[i] = static_cast<T>(start_from + static_cast<double>(random.Generate(k_range)) / k);
    }
}

/** @brief Fill a memory area with a sorted sequence of unique elements randomly generated.
 *
 *  This function generates and fills a blob of a certain precision, with a
 *  sorted sequence of unique elements.
 *
 * @param rawBlobDataPtr pointer to destination memory area
 * @param size number of elements in destination memory
 * @param range Values range
 * @param start_from Value from which range should start
 * @param k Resolution of floating point numbers.
 * - With k = 1 every random number will be basically integer number.
 * - With k = 2 numbers resolution will 1/2 so outputs only .0 or .50
 * - With k = 4 numbers resolution will 1/4 so outputs only .0 .25 .50 0.75 and etc.
 * @param seed seed of random generator
 */
template <typename T>
void inline fill_random_unique_sequence(T* rawBlobDataPtr,
                                        std::size_t size,
                                        uint64_t range,
                                        int64_t start_from = 0,
                                        const int64_t k = 1,
                                        const int32_t seed = 1) {
    if (start_from < 0 && !std::is_signed<T>::value) {
        start_from = 0;
    }

    if (range < size) {
        range = size * 2;
    }

    std::mt19937 generator(seed);
    std::uniform_int_distribution<int64_t> dist(k * start_from, k * (start_from + range));

    std::set<T> elems;
    while (elems.size() != size) {
        auto value = static_cast<float>(dist(generator));
        value /= static_cast<float>(k);
        if (std::is_same<ov::float16, T>::value) {
            elems.insert(static_cast<T>(ov::float16(value)));
        } else if (std::is_same<ov::bfloat16, T>::value) {
            elems.insert(static_cast<T>(ov::bfloat16(value)));
        } else {
            elems.insert(static_cast<T>(value));
        }
    }
    std::copy(elems.begin(), elems.end(), rawBlobDataPtr);
}

/** @brief Fill tensor with random data.
 *
 * @param tensor Target tensor
 * @param range Values range
 * @param start_from Value from which range should start
 * @param k Resolution of floating point numbers.
 * - With k = 1 every random number will be basically integer number.
 * - With k = 2 numbers resolution will 1/2 so outputs only .0 or .50
 * - With k = 4 numbers resolution will 1/4 so outputs only .0 .25 .50 0.75 and etc.
 */
void fill_tensor_random(ov::Tensor& tensor,
                        const double range = 10,
                        const double start_from = 0,
                        const int32_t k = 1,
                        const int seed = 1);

template <typename T>
void inline fill_data_ptr_consistently(T* data,
                                       size_t size,
                                       const uint32_t range = 10,
                                       int32_t start_from = 0,
                                       const int32_t k = 1) {
    int64_t value = start_from;
    const int64_t maxValue = start_from + range;
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(value);
        if (value < (maxValue - k)) {
            value += k;
        } else {
            value = start_from;
        }
    }
}

template <typename T>
void inline fill_data_ptr_normal_random_float(T* data,
                                              size_t size,
                                              const float mean,
                                              const float stddev,
                                              const int seed = 1) {
    std::default_random_engine random(seed);
    std::normal_distribution<> normal_d{mean, stddev};
    for (size_t i = 0; i < size; i++) {
        auto value = static_cast<float>(normal_d(random));
        if (typeid(T) == typeid(typename ov::fundamental_type_for<ov::element::f16>)) {
            data[i] = static_cast<T>(ov::float16(value).to_bits());
        } else {
            data[i] = static_cast<T>(value);
        }
    }
}

void fill_random_string(std::string* dst,
                        const size_t size,
                        const size_t len_range = 10lu,
                        const size_t start_from = 0lu,
                        const int seed = 1);

template <typename T>
typename std::enable_if<std::is_signed<T>::value, T>::type inline ie_abs(const T& val) {
    return std::abs(val);
}

template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, T>::type inline ie_abs(const T& val) {
    return val;
}

inline ov::bfloat16 ie_abs(const ov::bfloat16& val) {
    return ov::bfloat16::from_bits(val.to_bits() & 0x7FFF);
}

inline ov::float16 ie_abs(const ov::float16& val) {
    return ov::float16::from_bits(val.to_bits() & 0x7FFF);
}

inline ov::float8_e4m3 ie_abs(const ov::float8_e4m3& val) {
    return ov::float8_e4m3::from_bits(val.to_bits() & 0x7F);
}

inline ov::float8_e5m2 ie_abs(const ov::float8_e5m2& val) {
    return ov::float8_e5m2::from_bits(val.to_bits() & 0x7F);
}

template <class T_ACTUAL, class T_EXPECTED>
static void compare_raw_data(const T_EXPECTED* expected,
                             const T_ACTUAL* actual,
                             std::size_t size,
                             float threshold,
                             float abs_threshold = -1.f) {
    for (std::size_t i = 0; i < size; ++i) {
        const T_EXPECTED& ref = expected[i];
        const auto& res = actual[i];
        const auto absoluteDifference = ov::test::utils::ie_abs(res - ref);
        if (abs_threshold > 0.f && absoluteDifference > abs_threshold) {
            OPENVINO_THROW("Absolute comparison of values expected: ",
                           std::to_string(ref),
                           " and actual: ",
                           std::to_string(res),
                           " at index ",
                           i,
                           " with absolute threshold ",
                           abs_threshold,
                           " failed");
        }
        if (absoluteDifference <= threshold) {
            continue;
        }
        double max;
        if (sizeof(T_ACTUAL) < sizeof(T_EXPECTED)) {
            max = static_cast<double>(std::max(ov::test::utils::ie_abs(T_EXPECTED(res)), ov::test::utils::ie_abs(ref)));
        } else {
            max = static_cast<double>(std::max(ov::test::utils::ie_abs(res), ov::test::utils::ie_abs(T_ACTUAL(ref))));
        }
        double diff = static_cast<float>(absoluteDifference) / max;
        if (max == 0 || (diff > static_cast<float>(threshold)) ||
            (std::isnan(static_cast<float>(res)) ^ std::isnan(static_cast<float>(ref)))) {
            OPENVINO_THROW("Relative comparison of values expected: ",
                           std::to_string(ref),
                           " and actual: ",
                           std::to_string(res),
                           " at index ",
                           i,
                           " with threshold ",
                           threshold,
                           " failed");
        }
    }
}

}  // namespace utils
}  // namespace test
}  // namespace ov
