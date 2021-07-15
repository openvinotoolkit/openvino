// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <utility>

#include <gtest/gtest.h>
#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>

#include <ie_blob.h>
#include <random>

namespace CommonTestUtils {

inline void fill_data(float *data, size_t size, size_t duty_ratio = 10) {
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
void fill_data_with_broadcast(InferenceEngine::Blob::Ptr &blob, InferenceEngine::Blob::Ptr &values);

/**
 * Wrapper on top of fill_data_with_broadcast with simplified signature
 *
 * @param blob the destination blob to fill in
 * @param axis Axis to apply values
 * @param values data to broadcast
 */
void fill_data_with_broadcast(InferenceEngine::Blob::Ptr &blob, size_t axis, std::vector<float> values);

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
InferenceEngine::Blob::Ptr
make_reshape_view(const InferenceEngine::Blob::Ptr &blob, InferenceEngine::SizeVector new_shape);

/**
 * Fill blob with single value for all elements
 *
 * like:
 *     fill_data_with_broadcast(blob, 0, {val});
 *
 * @param blob tensor to fill in
 * @param val value to set into each element
 */
void fill_data_const(InferenceEngine::Blob::Ptr &blob, float val);


/**
 * Calculate size of buffer required for provided tensor descriptor.
 * @param tdesc provided tensor descriptor
 * @return size in bytes
 */
size_t byte_size(const InferenceEngine::TensorDesc &tdesc);

template<InferenceEngine::Precision::ePrecision PRC>
inline void
fill_data_roi(InferenceEngine::Blob::Ptr &blob, const uint32_t range, const int height, const int width, const float omega,
              const bool is_roi_max_mode, const int seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto *data = blob->buffer().as<dataType *>();
    std::default_random_engine random(seed);
    std::uniform_int_distribution<int32_t> distribution(0, range);

    const int max_y = (is_roi_max_mode) ? (height - 1) : 1;
    const int max_x = (is_roi_max_mode) ? (width - 1) : 1;

    float center_h = (max_y) / 2.0f;
    float center_w = (max_x) / 2.0f;

    for (size_t i = 0; i < blob->size(); i += 5) {
        data[i] = static_cast<dataType>(distribution(random));
        const float x0 = (center_w + width * 0.3f * sin(static_cast<float>(i + 1) * omega));
        const float x1 = (center_w + width * 0.3f * sin(static_cast<float>(i + 3) * omega));
        data[i + 1] = static_cast<dataType>(is_roi_max_mode ? std::floor(x0) : x0);
        data[i + 3] = static_cast<dataType>(is_roi_max_mode ? std::floor(x1) : x1);
        if (data[i + 3] < data[i + 1]) {
            std::swap(data[i + 1], data[i + 3]);
        }
        if (data[i + 1] < 0)
            data[i + 1] = 0;
        if (data[i + 3] > max_x)
            data[i + 3] = static_cast<dataType>(max_x);

        const float y0 = (center_h + height * 0.3f * sin(static_cast<float>(i + 2) * omega));
        const float y1 = (center_h + height * 0.3f * sin(static_cast<float>(i + 4) * omega));
        data[i + 2] = static_cast<dataType>(is_roi_max_mode ? std::floor(y0) : y0);
        data[i + 4] = static_cast<dataType>(is_roi_max_mode ? std::floor(y1) : y1);
        if (data[i + 4] < data[i + 2]) {
            std::swap(data[i + 2], data[i + 4]);
        }
        if (data[i + 2] < 0)
            data[i + 2] = 0;
        if (data[i + 4] > max_y)
            data[i + 4] = static_cast<dataType>(max_y);
    }
}

template<class T>
void inline
fill_data_random(T *pointer, std::size_t size, const uint32_t range = 10, int32_t start_from = 0, const int32_t k = 1,
                 const int seed = 1) {
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
void inline fill_data_random(InferenceEngine::Blob::Ptr &blob, const uint32_t range = 10, int32_t start_from = 0,
                             const int32_t k = 1, const int seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    fill_data_random(rawBlobDataPtr, blob->size(), range, start_from, k, seed);
}

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
template<InferenceEngine::Precision::ePrecision PRC>
void inline fill_random_unique_sequence(InferenceEngine::Blob::Ptr &blob,
                                        uint64_t range,
                                        int64_t start_from = 0,
                                        const int64_t k = 1,
                                        const int32_t seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();

    if (start_from < 0 && !std::is_signed<dataType>::value) {
        start_from = 0;
    }

    if (range < blob->size()) {
        range = blob->size() * 2;
    }

    std::mt19937 generator(seed);
    std::uniform_int_distribution<int64_t> dist(k * start_from, k * (start_from + range));

    std::set<dataType> elems;
    while (elems.size() != blob->size()) {
        auto value = static_cast<float>(dist(generator));
        value /= static_cast<float>(k);
        if (PRC == InferenceEngine::Precision::FP16) {
            elems.insert(static_cast<dataType>(ngraph::float16(value).to_bits()));
        } else {
            elems.insert(static_cast<dataType>(value));
        }
    }
    std::copy(elems.begin(), elems.end(), rawBlobDataPtr);
}

template<InferenceEngine::Precision::ePrecision PRC>
void inline fill_data_consistently(InferenceEngine::Blob::Ptr &blob, const uint32_t range = 10, int32_t start_from = 0,
                                   const int32_t k = 1) {
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
void inline
fill_data_random_float(InferenceEngine::Blob::Ptr &blob, const uint32_t range, int32_t start_from, const int32_t k,
                       const int seed = 1) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    std::default_random_engine random(seed);
    // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int32_t> distribution(k * start_from, k * (start_from + range));

    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    for (size_t i = 0; i < blob->size(); i++) {
        auto value = static_cast<float>(distribution(random));
        value /= static_cast<float>(k);
        if (PRC == InferenceEngine::Precision::FP16) {
            rawBlobDataPtr[i] = static_cast<dataType>(ngraph::float16(value).to_bits());
        } else if (PRC == InferenceEngine::Precision::BF16) {
            rawBlobDataPtr[i] = static_cast<dataType>(ngraph::bfloat16(value).to_bits());
        } else {
            rawBlobDataPtr[i] = static_cast<dataType>(value);
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
        if (typeid(dataType) ==
            typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = static_cast<dataType>(ngraph::float16(value).to_bits());
        } else {
            rawBlobDataPtr[i] = static_cast<dataType>(value);
        }
    }
}

template<InferenceEngine::Precision::ePrecision PRC, typename T>
void inline fill_data_float_array(InferenceEngine::Blob::Ptr &blob, const T values[], const size_t size) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;

    auto *rawBlobDataPtr = blob->buffer().as<dataType *>();
    for (size_t i = 0; i < std::min(size, blob->size()); i++) {
        auto value = values[i];
        if (typeid(dataType) ==
            typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = static_cast<dataType>(ngraph::float16(value).to_bits());

        } else {
            rawBlobDataPtr[i] = static_cast<dataType>(value);
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
inline ie_abs(const T &val) {
    return std::abs(val);
}

template<typename T>
typename std::enable_if<std::is_unsigned<T>::value, T>::type
inline ie_abs(const T &val) {
    return val;
}

inline ngraph::bfloat16 ie_abs(const ngraph::bfloat16 &val) {
    return ngraph::bfloat16::from_bits(val.to_bits() & 0x7FFF);
}

inline ngraph::float16 ie_abs(const ngraph::float16 &val) {
    return ngraph::float16::from_bits(val.to_bits() ^ 0x8000);
}

}  // namespace CommonTestUtils
