// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/identity.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_iterator.hpp"

template <ov::element::Type_t ET>
inline bool evaluate(const std::shared_ptr<ov::op::v15::Identity>& op,
                     ov::TensorVector& outputs,
                     const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
}

template <>
bool evaluate_node<ov::op::v15::Identity>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    const auto input_shape = inputs[0].get_shape();
    const auto total_elements =
        std::accumulate(input_shape.begin(), input_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    const auto total_size_in_bytes = get_memory_size(inputs[0].get_element_type(), total_elements);

    outputs[0].set_shape(input_shape);

    ov::reference::identity(static_cast<const char*>(inputs[0].data()),
                            static_cast<char*>(outputs[0].data()),
                            total_size_in_bytes);
    return true;
}
