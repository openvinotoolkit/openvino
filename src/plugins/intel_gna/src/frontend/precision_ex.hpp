// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_precision.hpp"

namespace InferenceEngine {

/**
 * @brief reverse trait for getting some precision from it's underlined memory type
 * this might not work for certain precisions : for Q78, U16
 * @tparam T
 */
template<class T>
struct precision_from_media {
    static const Precision::ePrecision type = Precision::CUSTOM;
};

template<>
struct precision_from_media<float> {
    static const Precision::ePrecision type = Precision::FP32;
};

template<>
struct precision_from_media<uint16_t> {
    static const Precision::ePrecision type = Precision::FP16;
};

template<>
struct precision_from_media<int16_t> {
    static const Precision::ePrecision type = Precision::I16;
};

template<>
struct precision_from_media<uint8_t> {
    static const Precision::ePrecision type = Precision::U8;
};

template<>
struct precision_from_media<int8_t> {
    static const Precision::ePrecision type = Precision::I8;
};

template<>
struct precision_from_media<int32_t> {
    static const Precision::ePrecision type = Precision::I32;
};

/**
 * @brief container for storing both precision and it's underlined media type
 * @tparam TMedia
 */
template <class TMedia>
class TPrecision : public Precision {
 public:
    typedef TMedia MediaType;
    TPrecision() : Precision(precision_from_media<TMedia>::type) {}
    explicit TPrecision(const Precision & that) : Precision(that) {}
    TPrecision & operator = (const Precision & that) {
        Precision::operator=(that);
        return *this;
    }
    explicit TPrecision(const Precision::ePrecision  value) : Precision(value) {}
};


// special case for Mixed, or undefined precisions
template <>
class TPrecision<void> : public Precision {
 public:
    typedef void MediaType;
    TPrecision() = default;
    explicit TPrecision(const Precision & that) : Precision(that) {}
    TPrecision & operator = (const Precision & that) {
        Precision::operator=(that);
        return *this;
    }
    explicit TPrecision(const Precision::ePrecision  value) : Precision(value) {}
};


}  // namespace InferenceEngine