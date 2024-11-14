// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/string_tensor_pack.hpp"

#include "evaluate_node.hpp"
#include "string_tensor_pack_shape_inference.hpp"

template <>
bool evaluate_node<ov::op::v15::StringTensorPack>(std::shared_ptr<ov::Node> node,
                                                  ov::TensorVector& outputs,
                                                  const ov::TensorVector& inputs) {
    auto string_tensor_pack = std::dynamic_pointer_cast<ov::op::v15::StringTensorPack>(node);
    OPENVINO_ASSERT(string_tensor_pack, "Node passed to StringTensorPack evaluate function is invalid.");
    ov::Shape output_shape;
    output_shape = ov::op::v15::shape_infer(string_tensor_pack.get(),
                                            ov::util::get_tensors_partial_shapes(inputs),
                                            make_tensor_accessor(inputs))
                       .front()
                       .to_shape();
    outputs.front().set_shape(output_shape);
    const auto indices_type = node->get_input_element_type(0);
    const auto& data_shape = node->get_input_shape(0);
    int64_t string_count = std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<int64_t>());
    switch (indices_type) {
    case ov::element::i32:
        ov::reference::string_tensor_pack(inputs[0].data<const int32_t>(),
                                          inputs[1].data<const int32_t>(),
                                          inputs[2].data<const uint8_t>(),
                                          outputs[0].data<std::string>(),
                                          string_count);
        break;
    case ov::element::i64:
        ov::reference::string_tensor_pack(inputs[0].data<const int64_t>(),
                                          inputs[1].data<const int64_t>(),
                                          inputs[2].data<const uint8_t>(),
                                          outputs[0].data<std::string>(),
                                          string_count);
        break;
    default:
        OPENVINO_THROW("Unhandled indices data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
    return true;
}
