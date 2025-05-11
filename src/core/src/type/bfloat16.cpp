// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Contains logic derived from TensorFlow’s bfloat16 implementation
// https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/bfloat16/bfloat16.h
// Copyright notice from original source file is as follows.

//*******************************************************************************
//  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//==============================================================================

#include "openvino/core/type/bfloat16.hpp"

#include <cmath>
#include <iostream>
#include <limits>

using namespace ov;

static_assert(sizeof(bfloat16) == 2, "class bfloat16 must be exactly 2 bytes");

inline bool float_isnan(const float& x) {
    return std::isnan(x);
}

std::vector<float> bfloat16::to_float_vector(const std::vector<bfloat16>& v_bf16) {
    std::vector<float> v_f32(v_bf16.begin(), v_bf16.end());
    return v_f32;
}

std::vector<bfloat16> bfloat16::from_float_vector(const std::vector<float>& v_f32) {
    std::vector<bfloat16> v_bf16;
    v_bf16.reserve(v_f32.size());
    for (float a : v_f32) {
        v_bf16.push_back(static_cast<bfloat16>(a));
    }
    return v_bf16;
}

std::string bfloat16::to_string() const {
    return std::to_string(static_cast<float>(*this));
}

size_t bfloat16::size() const {
    return sizeof(m_value);
}

#if defined __GNUC__ && __GNUC__ == 11
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuninitialized"
#endif

bfloat16::operator float() const {
    uint32_t tmp = 0;
    uint32_t* ptmp = &tmp;
    *ptmp = (static_cast<uint32_t>(m_value) << 16);
    const float* f = reinterpret_cast<const float*>(ptmp);
    return *f;
}

#if defined __GNUC__ && __GNUC__ == 11
#    pragma GCC diagnostic pop
#endif

uint16_t bfloat16::to_bits() const {
    return m_value;
}
