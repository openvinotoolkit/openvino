// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "tensor_lite_place.hpp"

#include "quantization_info.hpp"

void ov::frontend::tensorflow_lite::TensorLitePlace::translate(ov::Output<ov::Node>& output,
                                                               bool convert_tensor_attrs_to_nodes) {
    output.set_names({*get_names().begin()});
    output.get_rt_info()[QuantizationInfo::get_type_info_static()] = m_quantization;
    if (convert_tensor_attrs_to_nodes)
        apply_quantization(output);
}
