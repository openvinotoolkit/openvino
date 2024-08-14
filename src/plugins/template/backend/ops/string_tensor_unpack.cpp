// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/string_tensor_unpack.hpp"

#include "evaluate_node.hpp"
#include "string_tensor_unpack_shape_inference.hpp"

template <>
bool evaluate_node<ov::op::v15::StringTensorUnpack>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs) {
    if (node->get_input_element_type(0) == ov::element::string) {
        auto string_tensor_unpack = std::dynamic_pointer_cast<ov::op::v15::StringTensorUnpack>(node);
        OPENVINO_ASSERT(string_tensor_unpack, "Node passed to StringTensorUnpack evaluate function is invalid.");
        std::vector<ov::PartialShape> output_shapes;
        output_shapes = ov::op::v15::shape_infer(string_tensor_unpack.get(),
                                                 ov::util::get_tensors_partial_shapes(inputs),
                                                 make_tensor_accessor(inputs));
        auto outputs_it = outputs.begin();
        for (const auto& p_shape : output_shapes) {
            outputs_it->set_shape(p_shape.get_shape());
            ++outputs_it;
        }
        const auto& data_shape = node->get_input_shape(0);
        const auto element_count = shape_size(data_shape);
        ov::reference::string_tensor_unpack(inputs[0].data<const std::string>(),
                                            outputs[0].data<int32_t>(),
                                            outputs[1].data<int32_t>(),
                                            outputs[2].data<uint8_t>(),
                                            element_count);
    } else {
        OPENVINO_THROW("The input type for StringTensorUnpack operation must be ov::element::string.");
    }
    return true;
}
