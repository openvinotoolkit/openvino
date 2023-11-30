// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include "common_test_utils/common_utils.hpp"
#include "gtest/gtest.h"
#include "ie_blob.h"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/tensor.hpp"

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
 * Fill blob with value data blob. Broadcast semantic is included.
 * Broadcasting with alignment through last dimension.
 *
 * @param blob tensor to fill in
 * @param values src tensor which should be broadcast
 */
OPENVINO_SUPPRESS_DEPRECATED_START
void fill_data_with_broadcast(InferenceEngine::Blob::Ptr& blob, InferenceEngine::Blob::Ptr& values);
OPENVINO_SUPPRESS_DEPRECATED_END
void fill_data_with_broadcast(ov::Tensor& tensor, ov::Tensor& values);

/**
 * Wrapper on top of fill_data_with_broadcast with simplified signature
 *
 * @param blob the destination blob to fill in
 * @param axis Axis to apply values
 * @param values data to broadcast
 */
OPENVINO_SUPPRESS_DEPRECATED_START
void fill_data_with_broadcast(InferenceEngine::Blob::Ptr& blob, size_t axis, std::vector<float> values);
OPENVINO_SUPPRESS_DEPRECATED_END
void fill_data_with_broadcast(ov::Tensor& tensor, size_t axis, std::vector<float> values);
/**
 * Make a view blob with new shape. It will reinterpret original tensor data as a tensor with new shape.
 *
 * NB! Limitation: the nwe one blob will no have ownership of data buffer. The original blob should be alive
 *     while view is in use.
 *
 * @param tensor original source tensor
 * @param new_shape new one shape for view blob
 * @return new one blob view
 */
OPENVINO_SUPPRESS_DEPRECATED_START
InferenceEngine::Blob::Ptr make_reshape_view(const InferenceEngine::Blob::Ptr& blob,
                                             InferenceEngine::SizeVector new_shape);
OPENVINO_SUPPRESS_DEPRECATED_END

/**
 * Calculate size of buffer required for provided tensor descriptor.
 * @param tdesc provided tensor descriptor
 * @return size in bytes
 */
OPENVINO_SUPPRESS_DEPRECATED_START
size_t byte_size(const InferenceEngine::TensorDesc& tdesc);
OPENVINO_SUPPRESS_DEPRECATED_END

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

OPENVINO_SUPPRESS_DEPRECATED_START
template <InferenceEngine::Precision::ePrecision PRC>
inline void fill_data_roi(InferenceEngine::Blob::Ptr& blob,
                          const uint32_t range,
                          const int height,
                          const int width,
                          const float omega,
                          const bool is_roi_max_mode,
                          const int seed = 1,
                          void (*propGenerator)(InferenceEngine::Blob::Ptr&) = nullptr) {
    if (propGenerator != nullptr) {
        propGenerator(blob);
        return;
    }
    using T = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto* data = blob->buffer().as<T*>();
    fill_roi_raw_ptr<T>(data, blob->size(), range, height, width, omega, is_roi_max_mode, seed);
}

void fill_data_roi(ov::runtime::Tensor& tensor,
                   const uint32_t range,
                   const int height,
                   const int width,
                   const float omega,
                   const bool is_roi_max_mode,
                   const int seed = 1);

OPENVINO_SUPPRESS_DEPRECATED_END

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

/** @brief Fill blob with random data.
 *
 * @param blob Target blob
 * @param range Values range
 * @param start_from Value from which range should start
 * @param k Resolution of floating point numbers.
 * - With k = 1 every random number will be basically integer number.
 * - With k = 2 numbers resolution will 1/2 so outputs only .0 or .50
 * - With k = 4 numbers resolution will 1/4 so outputs only .0 .25 .50 0.75 and etc.
 */
OPENVINO_SUPPRESS_DEPRECATED_START
template <InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_random(InferenceEngine::Blob::Ptr& blob,
                             const uint32_t range = 10,
                             int32_t start_from = 0,
                             const int32_t k = 1,
                             const int seed = 1) {
    using T = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto* rawBlobDataPtr = blob->buffer().as<T*>();
    if (PRC == InferenceEngine::Precision::U4 || PRC == InferenceEngine::Precision::I4 ||
        PRC == InferenceEngine::Precision::BIN) {
        fill_data_random(rawBlobDataPtr, blob->byteSize(), range, start_from, k, seed);
    } else {
        fill_data_random(rawBlobDataPtr, blob->size(), range, start_from, k, seed);
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END

/** @brief Fill blob with a sorted sequence of unique elements randomly generated.
 *
 *  This function generates and fills a blob of a certain precision, with a
 *  sorted sequence of unique elements.
 *
 * @param blob Target blob
 * @param range Values range
 * @param start_from Value from which range should start
 * @param k Resolution of floating point numbers.
 * - With k = 1 every random number will be basically integer number.
 * - With k = 2 numbers resolution will 1/2 so outputs only .0 or .50
 * - With k = 4 numbers resolution will 1/4 so outputs only .0 .25 .50 0.75 and etc.
 */
OPENVINO_SUPPRESS_DEPRECATED_START
template <InferenceEngine::Precision::ePrecision PRC>
void inline fill_random_unique_sequence(InferenceEngine::Blob::Ptr& blob,
                                        uint64_t range,
                                        int64_t start_from = 0,
                                        const int64_t k = 1,
                                        const int32_t seed = 1) {
    using T = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto* rawBlobDataPtr = blob->buffer().as<T*>();

    if (start_from < 0 && !std::is_signed<T>::value) {
        start_from = 0;
    }

    if (range < blob->size()) {
        range = blob->size() * 2;
    }

    std::mt19937 generator(seed);
    std::uniform_int_distribution<int64_t> dist(k * start_from, k * (start_from + range));

    std::set<T> elems;
    while (elems.size() != blob->size()) {
        auto value = static_cast<float>(dist(generator));
        value /= static_cast<float>(k);
        if (PRC == InferenceEngine::Precision::FP16) {
            elems.insert(static_cast<T>(ov::float16(value).to_bits()));
        } else {
            elems.insert(static_cast<T>(value));
        }
    }
    std::copy(elems.begin(), elems.end(), rawBlobDataPtr);
}
OPENVINO_SUPPRESS_DEPRECATED_END

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

OPENVINO_SUPPRESS_DEPRECATED_START
template <InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_consistently(InferenceEngine::Blob::Ptr& blob,
                                   const uint32_t range = 10,
                                   int32_t start_from = 0,
                                   const int32_t k = 1) {
    using T = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto* rawBlobDataPtr = blob->buffer().as<T*>();
    if (start_from < 0 && !std::is_signed<T>::value) {
        start_from = 0;
    }
    fill_data_ptr_consistently(rawBlobDataPtr, blob->size(), range, start_from, k);
}

template <InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_random_float(InferenceEngine::Blob::Ptr& blob,
                                   const uint32_t range,
                                   int32_t start_from,
                                   const int32_t k,
                                   const int seed = 1) {
    using T = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    std::default_random_engine random(seed);
    // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int32_t> distribution(k * start_from, k * (start_from + range));

    auto* rawBlobDataPtr = blob->buffer().as<T*>();
    for (size_t i = 0; i < blob->size(); i++) {
        auto value = static_cast<float>(distribution(random));
        value /= static_cast<float>(k);
        if (PRC == InferenceEngine::Precision::FP16) {
            rawBlobDataPtr[i] = static_cast<T>(ov::float16(value).to_bits());
        } else if (PRC == InferenceEngine::Precision::BF16) {
            rawBlobDataPtr[i] = static_cast<T>(ov::bfloat16(value).to_bits());
        } else {
            rawBlobDataPtr[i] = static_cast<T>(value);
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
        if (typeid(T) ==
            typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            data[i] = static_cast<T>(ov::float16(value).to_bits());
        } else {
            data[i] = static_cast<T>(value);
        }
    }
}

template <InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_normal_random_float(InferenceEngine::Blob::Ptr& blob,
                                          const float mean,
                                          const float stddev,
                                          const int seed = 1) {
    using T = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto* rawBlobDataPtr = blob->buffer().as<T*>();
    fill_data_ptr_normal_random_float<T>(rawBlobDataPtr, blob->size(), mean, stddev, seed);
}

template <InferenceEngine::Precision::ePrecision PRC, typename T>
void inline fill_data_float_array(InferenceEngine::Blob::Ptr& blob, const T values[], const size_t size) {
    using Type = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    auto* rawBlobDataPtr = blob->buffer().as<T*>();
    for (size_t i = 0; i < std::min(size, blob->size()); i++) {
        auto value = values[i];
        if (typeid(Type) ==
            typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = static_cast<Type>(ov::float16(value).to_bits());

        } else {
            rawBlobDataPtr[i] = static_cast<Type>(value);
        }
    }
}

template <>
void inline fill_data_random<InferenceEngine::Precision::FP32>(InferenceEngine::Blob::Ptr& blob,
                                                               const uint32_t range,
                                                               int32_t start_from,
                                                               const int32_t k,
                                                               const int seed) {
    fill_data_random_float<InferenceEngine::Precision::FP32>(blob, range, start_from, k, seed);
}

template <>
void inline fill_data_random<InferenceEngine::Precision::FP16>(InferenceEngine::Blob::Ptr& blob,
                                                               const uint32_t range,
                                                               int32_t start_from,
                                                               const int32_t k,
                                                               const int seed) {
    fill_data_random_float<InferenceEngine::Precision::FP16>(blob, range, start_from, k, seed);
}

template <>
void inline fill_data_random<InferenceEngine::Precision::BF16>(InferenceEngine::Blob::Ptr& blob,
                                                               const uint32_t range,
                                                               int32_t start_from,
                                                               const int32_t k,
                                                               const int seed) {
    fill_data_random_float<InferenceEngine::Precision::BF16>(blob, range, start_from, k, seed);
}
OPENVINO_SUPPRESS_DEPRECATED_END

inline void fill_random_string(std::string* dst,
                               const size_t size,
                               const size_t len_range = 10lu,
                               const size_t start_from = 0lu,
                               const int seed = 1) {
    static const int32_t char_range = 128;
    testing::internal::Random random_len(seed);
    random_len.Generate(len_range);
    testing::internal::Random random_char(seed);
    random_char.Generate(char_range);


    for (size_t i = 0lu; i < size; i++) {
        const auto len = start_from + static_cast<size_t>(random_len.Generate(len_range));
        auto& str = dst[i];
        str.resize(len);
        for (size_t j = 0lu; j < len; j++) {
            str[j] = static_cast<char>(random_len.Generate(char_range));
        }
    }
}

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

}  // namespace utils
}  // namespace test
}  // namespace ov
