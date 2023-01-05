// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "tensor_lite_place.hpp"

#include "quantization_info.hpp"

std::shared_ptr<ov::frontend::tensorflow_lite::Quantization>
ov::frontend::tensorflow_lite::TensorLitePlace::get_quantization() const {
    return m_quantization;
}

void ov::frontend::tensorflow_lite::TensorLitePlace::disable_quantization() {
    m_quantization->no_quantization = true;
}

void ov::frontend::tensorflow_lite::TensorLitePlace::translate(ov::Output<ov::Node>& output,
                                                               bool convert_tensor_attrs_to_nodes) {
    output.set_names({*get_names().begin()});
    output.get_rt_info()[QuantizationInfo::get_type_info_static()] = QuantizationInfo(get_quantization());
}
