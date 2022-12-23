// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils.hpp"
#include "tensor_lite_place.hpp"

std::shared_ptr<ov::frontend::tensorflow_lite::Quantization> ov::frontend::tensorflow_lite::TensorLitePlace::get_quantization() const {
    return m_quantization;
}

void ov::frontend::tensorflow_lite::TensorLitePlace::disable_quantization() {
    m_quantization->no_quantization = true;
}

void ov::frontend::tensorflow_lite::TensorLitePlace::translate(ov::Output<ov::Node> &output) {
    // output = apply_quantization(output, this);
    output.set_names({*get_names().begin()});
}
