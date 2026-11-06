// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gguf_dequantize.hpp"

#include "evaluate_node.hpp"
#include "gguf_dequantize_shape_inference.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/gguf_dequantize.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::GGUFDequantize>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shapes = ov::op::internal::shape_infer(op.get(), input_shapes);
    outputs[0].set_shape(output_shapes[0].to_shape());

    using Format = ov::op::internal::GGUFDequantize::Format;
    const auto format = op->get_format() == Format::Q4_K   ? ov::reference::gguf::Format::Q4_K
                        : op->get_format() == Format::Q5_K ? ov::reference::gguf::Format::Q5_K
                                                           : ov::reference::gguf::Format::Q6_K;
    ov::reference::gguf_dequantize(inputs[0].data<const uint8_t>(),
                                   outputs[0].data<T>(),
                                   inputs[0].get_byte_size() / ov::reference::gguf::block_bytes(format),
                                   format);
    return true;
}

template <>
bool evaluate_node<ov::op::internal::GGUFDequantize>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::internal::GGUFDequantize>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::internal::GGUFDequantize>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::internal::GGUFDequantize>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
