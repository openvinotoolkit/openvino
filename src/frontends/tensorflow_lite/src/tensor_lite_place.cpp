// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils.hpp"
#include "tensor_lite_place.hpp"

const ov::frontend::tensorflow_lite::Quantization& ov::frontend::tensorflow_lite::TensorLitePlace::get_quantization() const {
    return m_quantization;
}

