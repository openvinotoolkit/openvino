// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_lite_place.hpp"

#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "utils.hpp"

void ov::frontend::tensorflow_lite::TensorLitePlace::translate(ov::Output<ov::Node>& output,
                                                               bool convert_tensor_attrs_to_nodes) {
    output.set_names({*get_names().begin()});
    output.get_rt_info()[ov::frontend::tensorflow_lite::QuantizationInfo::get_type_info_static()] = m_quantization;
    output.get_rt_info()[ov::frontend::tensorflow_lite::SparsityInfo::get_type_info_static()] = m_sparsity;
    if (convert_tensor_attrs_to_nodes)
        apply_quantization(output, get_element_type());
}
