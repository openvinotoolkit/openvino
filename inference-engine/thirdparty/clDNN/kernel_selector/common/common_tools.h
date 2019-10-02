// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once
#include "common_types.h"
#include <type_traits>

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
        case Datatype::BINARY:
            return 4;
        case Datatype::INT64:
            return 8;
        default:
            return 0;
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
        case WeightsType::BINARY:
            return 4;
        default:
            return 0;
    }
}

inline uint8_t GetActivationAdditionalParamsNumber(ActivationFunction func) {
    uint8_t paramsNum = 0;

    switch (func) {
        case ActivationFunction::LINEAR:
        case ActivationFunction::CLAMP:
            paramsNum = 2;
            break;
        case ActivationFunction::RELU_NEGATIVE_SLOPE:
        case ActivationFunction::ELU:
        case ActivationFunction::RELU_NEGATIVE_SLOPE_GRAD:
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
