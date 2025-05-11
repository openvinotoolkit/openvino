// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "common_types.h"
#include <type_traits>
#include <stdexcept>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BytesPerElement
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline uint32_t BytesPerElement(Datatype dt) {
    switch (dt) {
        case Datatype::INT8:
        case Datatype::UINT8:
            return 1;
        case Datatype::F16:
        case Datatype::INT16:
        case Datatype::UINT16:
            return 2;
        case Datatype::F32:
        case Datatype::INT32:
        case Datatype::UINT32:
            return 4;
        case Datatype::INT64:
            return 8;
        default:
            throw std::runtime_error("[GPU] BytesPerElement doesn't support given precision");
    }
}

inline uint32_t BytesPerElement(WeightsType wt) {
    switch (wt) {
        case WeightsType::INT8:
        case WeightsType::UINT8:
            return 1;
        case WeightsType::F16:
            return 2;
        case WeightsType::F32:
        case WeightsType::INT32:
            return 4;
        default:
            throw std::runtime_error("[GPU] BytesPerElement doesn't support given precision");
    }
}

inline uint8_t GetActivationAdditionalParamsNumber(ActivationFunction func) {
    uint8_t paramsNum = 0;

    switch (func) {
        case ActivationFunction::LINEAR:
        case ActivationFunction::CLAMP:
        case ActivationFunction::HARD_SIGMOID:
        case ActivationFunction::SELU:
            paramsNum = 2;
            break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE:
        case ActivationFunction::ELU:
        case ActivationFunction::POW:
        case ActivationFunction::SWISH:
            paramsNum = 1;
            break;
        default:
            break;
    }

    return paramsNum;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type Align(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? size : size - size % align + align);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type Pad(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? 0 : align - size % align);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type IsAligned(T size, size_t align) {
    return !(size % align);
}

template <typename T1, typename T2>
constexpr auto CeilDiv(T1 val, T2 divider) ->
    typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
                            decltype(std::declval<typename std::make_unsigned<T1>::type>() /
                                     std::declval<typename std::make_unsigned<T2>::type>())>::type {
    typedef typename std::make_unsigned<T1>::type UT1;
    typedef typename std::make_unsigned<T2>::type UT2;
    typedef decltype(std::declval<UT1>() / std::declval<UT2>()) RetT;

    return static_cast<RetT>((static_cast<UT1>(val) + static_cast<UT2>(divider) - 1U) / static_cast<UT2>(divider));
}

template <typename T1, typename T2>
constexpr auto RoundUp(T1 val, T2 rounding) ->
    typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
                            decltype(std::declval<typename std::make_unsigned<T1>::type>() /
                                     std::declval<typename std::make_unsigned<T2>::type>())>::type {
    typedef typename std::make_unsigned<T1>::type UT1;
    typedef typename std::make_unsigned<T2>::type UT2;
    typedef decltype(std::declval<UT1>() / std::declval<UT2>()) RetT;

    return static_cast<RetT>(CeilDiv(val, rounding) * static_cast<UT2>(rounding));
}
}  // namespace kernel_selector
