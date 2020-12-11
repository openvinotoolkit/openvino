// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cmath>
#include <utility>

#include <gtest/gtest.h>
#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>

#include <ie_blob.h>
#include <blob_factory.hpp>
#include <random>

namespace CommonTestUtils {

static void fill_data(float *data, size_t size, size_t duty_ratio = 10) {
    for (size_t i = 0; i < size; i++) {
        if ((i / duty_ratio) % 2 == 1) {
            data[i] = 0.0f;
        } else {
            data[i] = sin(static_cast<float>(i));
        }
    }
}

static void fill_data_sine(float *data, size_t size, float center, float ampl, float omega) {
    for (size_t i = 0; i < size; i++) {
        data[i] = center + ampl * sin(static_cast<float>(i) * omega);
    }
}

/**
 * @brief Create vector of floats with length of vec_len, with values ranging from min to max, 
 * with initial seed equal to variable seed with default of 0
 */
static inline std::vector<float> generate_float_numbers(std::size_t vec_len, float min, float max, int seed = 0) {
    std::vector<float> res;
    std::mt19937 gen(static_cast<float>(seed));

    std::uniform_real_distribution<float> dist(min, max);
    for (int i = 0; i < vec_len; i++)
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
void fill_data_with_broadcast(InferenceEngine::Blob::Ptr& blob, InferenceEngine::Blob::Ptr& values);

/**
 * Wrapper on top of fill_data_with_broadcast with simplified signature
 *
 * @param blob the destination blob to fill in
 * @param axis Axis to apply values
 * @param values data to broadcast
 */
void fill_data_with_broadcast(InferenceEngine::Blob::Ptr& blob, size_t axis, std::vector<float> values);

/**
 * Make a view blob with new shape. It will reinterpret original tensor data as a tensor with new shape.
 *
 * NB! Limitation: the nwe one blob will no have ownership of data buffer. The original blob should be alive
 *     while view is in use.
 *
 * @param blob original source tensor
 * @param new_shape new one shape for view blob
 * @return new one blob view
 */
InferenceEngine::Blob::Ptr make_reshape_view(const InferenceEngine::Blob::Ptr &blob, InferenceEngine::SizeVector new_shape);

/**
 * Fill blob with single value for all elements
 *
 * like:
 *     fill_data_with_broadcast(blob, 0, {val});
 *
 * @param blob tensor to fill in
 * @param val value to set into each element
 */
void fill_data_const(InferenceEngine::Blob::Ptr& blob, float val);


/**
 * Calculate size of buffer required for provided tensor descriptor.
 * @param tdesc provided tensor descriptor
 * @return size in bytes
 */
size_t byte_size(const InferenceEngine::TensorDesc &tdesc);

static void fill_data_bbox(float *data, size_t size, int height, int width, float omega) {
    float center_h = (height - 1.0f) / 2;
    float center_w = (width - 1.0f) / 2;
    for (size_t i = 0; i < size; i = i + 5) {
        data[i] = 0.0f;
        data[i + 1] = center_w + width * 0.6f * sin(static_cast<float>(i+1) * omega);
        data[i + 3] = center_w + width * 0.6f * sin(static_cast<float>(i+3) * omega);
        if (data[i + 3] < data[i + 1]) {
            std::swap(data[i + 1], data[i + 3]);
        }
        if (data[i + 1] < 0)
            data[i + 1] = 0;
        if (data[i + 3] > width - 1)
            data[i + 3] = static_cast<float>(width - 1);

        data[i + 2] = center_h + height * 0.6f * sin(static_cast<float>(i+2) * omega);
        data[i + 4] = center_h + height * 0.6f * sin(static_cast<float>(i+4) * omega);
        if (data[i + 4] < data[i + 2]) {
            std::swap(data[i + 2], data[i + 4]);
        }
        if (data[i + 2] < 0)
            data[i + 2] = 0;
        if (data[i + 4] > height - 1)
            data[i + 4] = static_cast<float>(height - 1);
    }
}

static void fill_data_roi(float *data, size_t size, const uint32_t range, const int height, const int width, const float omega, const int seed = 1) {
    std::default_random_engine random(seed);
    std::uniform_int_distribution<int32_t> distribution(0, range);
    float center_h = (height - 1.0f) / 2;
    float center_w = (width - 1.0f) / 2;
    for (size_t i = 0; i < size; i += 5) {
        data[i] = static_cast<float>(distribution(random));
        data[i + 1] = std::floor(center_w + width * 0.6f * sin(static_cast<float>(i+1) * omega));
        data[i + 3] = std::floor(center_w + width * 0.6f * sin(static_cast<float>(i+3) * omega));
        if (data[i + 3] < data[i + 1]) {
            std::swap(data[i + 1], data[i + 3]);
        }
        if (data[i + 1] < 0)
            data[i + 1] = 0;
        if (data[i + 3] > width - 1)
            data[i + 3] = static_cast<float>(width - 1);

        data[i + 2] = std::floor(center_h + height * 0.6f * sin(static_cast<float>(i+2) * omega));
        data[i + 4] = std::floor(center_h + height * 0.6f * sin(static_cast<float>(i+4) * omega));
        if (data[i + 4] < data[i + 2]) {
            std::swap(data[i + 2], data[i + 4]);
        }
        if (data[i + 2] < 0)
            data[i + 2] = 0;
        if (data[i + 4] > height - 1)
            data[i + 4] = static_cast<float>(height - 1);
    }
}

template<class T>
void inline fill_data_random(T* pointer, std::size_t size, const uint32_t range = 10, int32_t start_from = 0, const int32_t k = 1, const int seed = 1) {
    testing::internal::Random random(seed);
    random.Generate(range);

    if (start_from < 0 && !std::is_signed<T>::value) {
        start_from = 0;
    }

    for (std::size_t i = 0; i < size; i++) {
        pointer[i] = static_cast<T>(start_from + static_cast<int64_t>(random.Generate(range)));
    }
}

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
template<InferenceEngine::Precision::ePrecision PRC>
void inline  fill_data_random(InferenceEngine::Blob::Ptr &blob, const uint32_t range = 10, int32_t start_from = 0, const int32_t k = 1, const int seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    fill_data_random(rawBlobDataPtr, blob->size(), range, start_from, k, seed);
}

template<InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_consistently(InferenceEngine::Blob::Ptr &blob, const uint32_t range = 10, int32_t start_from = 0, const int32_t k = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    if (start_from < 0 && !std::is_signed<dataType>::value) {
        start_from = 0;
    }

    int64_t value = start_from;
    const int64_t maxValue = start_from + range;
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = static_cast<dataType>(value);
        if (value < (maxValue - k)) {
            value += k;
        } else {
            value = start_from;
        }
    }
}

template<InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_random_float(InferenceEngine::Blob::Ptr &blob, const uint32_t range, int32_t start_from, const int32_t k, const int seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    std::default_random_engine random(seed);
    // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int32_t> distribution(k * start_from, k * (start_from + range));

    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    for (size_t i = 0; i < blob->size(); i++) {
        auto value = static_cast<float>(distribution(random));
        value /= static_cast<float>(k);
        if (PRC == InferenceEngine::Precision::FP16) {
            rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
        } else if (PRC == InferenceEngine::Precision::BF16) {
            rawBlobDataPtr[i] = ngraph::bfloat16(value).to_bits();
        } else {
            rawBlobDataPtr[i] = value;
        }
    }
}

template<InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_normal_random_float(InferenceEngine::Blob::Ptr &blob,
                                          const float mean,
                                          const float stddev,
                                          const int seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    std::default_random_engine random(seed);
    std::normal_distribution<> normal_d{mean, stddev};

    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    for (size_t i = 0; i < blob->size(); i++) {
        auto value = static_cast<float>(normal_d(random));
        if (typeid(dataType) == typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
        } else {
            rawBlobDataPtr[i] = value;
        }
    }
}

template<InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_float_array(InferenceEngine::Blob::Ptr &blob, const float values[], const size_t size) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    for (size_t i = 0; i < std::min(size, blob->size()); i++) {
        auto value = values[i];
        if (typeid(dataType) == typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
        } else {
            rawBlobDataPtr[i] = value;
        }
    }
}

template<>
void inline fill_data_random<InferenceEngine::Precision::FP32>(InferenceEngine::Blob::Ptr &blob,
                                                               const uint32_t range,
                                                               int32_t start_from,
                                                               const int32_t k,
                                                               const int seed) {
    fill_data_random_float<InferenceEngine::Precision::FP32>(blob, range, start_from, k, seed);
}

template<>
void inline fill_data_random<InferenceEngine::Precision::FP16>(InferenceEngine::Blob::Ptr &blob,
                                                               const uint32_t range,
                                                               int32_t start_from,
                                                               const int32_t k, const int seed) {
    fill_data_random_float<InferenceEngine::Precision::FP16>(blob, range, start_from, k, seed);
}

template<>
void inline fill_data_random<InferenceEngine::Precision::BF16>(InferenceEngine::Blob::Ptr &blob,
                                                               const uint32_t range,
                                                               int32_t start_from,
                                                               const int32_t k, const int seed) {
    fill_data_random_float<InferenceEngine::Precision::BF16>(blob, range, start_from, k, seed);
}

template<typename T>
typename std::enable_if<std::is_signed<T>::value, T>::type
static ie_abs(const T &val) {
    return std::abs(val);
}

template<typename T>
typename std::enable_if<std::is_unsigned<T>::value, T>::type
static ie_abs(const T &val) {
    return val;
}

static ngraph::bfloat16 ie_abs(const ngraph::bfloat16& val) {
    return ngraph::bfloat16::from_bits(val.to_bits() ^ 0x8000);
}
}  // namespace CommonTestUtils
