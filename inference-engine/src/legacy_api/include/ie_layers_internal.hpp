// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <tuple>

#include "ie_api.h"
#include "ie_layers.h"
#include "ie_util_internal.hpp"

namespace InferenceEngine {

class Paddings {
public:
    PropertyVector<unsigned int> begin;
    PropertyVector<unsigned int> end;
};

/**
 * @brief gets padding with runtime type check
 */
INFERENCE_ENGINE_API_CPP(Paddings) getPaddingsImpl(const CNNLayer& layer);

/**
 * @brief checks that given type is one of specified in variadic template list
 * @tparam ...
 */
template <typename...>
struct is_one_of {
    static constexpr bool value = false;
};

/**
 * @brief checks that given type is one of specified in variadic template list
 * @tparam ...
 */
template <typename F, typename S, typename... T>
struct is_one_of<F, S, T...> {
    static constexpr bool value = std::is_same<F, S>::value || is_one_of<F, T...>::value;
};

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief gets padding without compile-time type check
 */
template <class T>
inline typename std::enable_if<is_one_of<T, DeformableConvolutionLayer, DeconvolutionLayer, ConvolutionLayer,
                                         BinaryConvolutionLayer, PoolingLayer>::value,
                               Paddings>::type
getPaddings(const T& layer) {
    return getPaddingsImpl(layer);
}

/*********************************************
 * TensorIterator Helpers section
 *********************************************/

/**
 * @brief Calculate number of iteration required for provided TI layer
 *
 * @param ti TensorIterator layer to parse
 * @return positive value in case of correct TI layer, -1 in case of inconsistency
 */
INFERENCE_ENGINE_API_CPP(int) getNumIteration(const TensorIterator& ti);

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
