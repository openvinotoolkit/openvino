// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <tuple>
#include "ie_api.h"
#include "ie_layers.h"
#include "ie_util_internal.hpp"

namespace InferenceEngine {

class INFERENCE_ENGINE_API_CLASS(Paddings) {
public:
    PropertyVector<unsigned int> begin;
    PropertyVector<unsigned int> end;
};

/**
 * @brief gets padding with runtime type check
 */
INFERENCE_ENGINE_API_CPP(Paddings) getPaddingsImpl(const CNNLayer &layer);

/**
 * @brief gets padding without compile-time type check
 */
template <class T>
inline  typename std::enable_if<is_one_of<T,
                                          DeconvolutionLayer,
                                          ConvolutionLayer,
                                          PoolingLayer>::value, Paddings>::type
getPaddings(const T & layer) {
    return getPaddingsImpl(layer);
}


}  // namespace InferenceEngine
