// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_conversion_util.hpp"

#include "openvino/core/shape_util.hpp"

namespace ov {
namespace util {
OPENVINO_SUPPRESS_DEPRECATED_START
Tensor wrap_tensor(const ngraph::HostTensorPtr& t) {
    const auto& et = t->get_element_type();
    const auto& p_shape = t->get_partial_shape();

    if (et.is_dynamic() || et == element::undefined) {
        return {};
    } else if (p_shape.is_static()) {
        return {et, p_shape.to_shape(), t->get_data_ptr()};
    } else {
        return {et, Shape{0}};
    }
}

Tensor wrap_tensor(const Output<Node>& output) {
    const auto& et = output.get_element_type();
    const auto& p_shape = output.get_partial_shape();

    if (et.is_dynamic() || et == element::undefined) {
        return {};
    } else if (p_shape.is_static()) {
        return {et, p_shape.to_shape()};
    } else {
        return {et, Shape{0}};
    }
}

ov::TensorVector wrap_tensors(const std::vector<ngraph::HostTensorPtr>& tensors) {
    ov::TensorVector out;
    out.reserve(tensors.size());
    for (const auto& ht : tensors) {
        out.push_back(ov::util::wrap_tensor(ht));
    }
    return out;
}

void update_output_host_tensors(const std::vector<ngraph::HostTensorPtr>& output_values,
                                const ov::TensorVector& outputs) {
    OPENVINO_ASSERT(output_values.size() == outputs.size());
    for (size_t i = 0; i < output_values.size(); ++i) {
        auto& ht = output_values[i];
        auto& t = outputs[i];
        if (ht->get_partial_shape().is_dynamic()) {
            ht->set_element_type(t.get_element_type());
            ht->set_shape(t.get_shape());
            std::memcpy(ht->get_data_ptr(), t.data(), t.get_byte_size());
        }
    }
}
OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace util
}  // namespace ov
