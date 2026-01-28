// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov {
namespace util {
/**
 * @brief Creates default tensor name for given Node's output.
 * The name format is "node_name:output_port".
 *
 * @param output - Node's output to create name for tensor.
 * @return Default tensor name.
 */
OPENVINO_API std::string make_default_tensor_name(const Output<const Node>& output);
}  // namespace util

namespace op::util {
/**
 * @brief Set name for both node and output tensor. Any other names will be overriden by a given single name
 * @param node - node to rename
 * @param name - new name
 * @param output_port - output port to rename
 */
void OPENVINO_API set_name(ov::Node& node, const std::string& name, size_t output_port = 0);


class ConstantDescriptor {
public:
    static ConstantDescriptor get_desc(const std::shared_ptr<ov::op::v0::Constant>& constant) {
        ConstantDescriptor desc;
        auto itag_buffer = std::dynamic_pointer_cast<ov::ITagBuffer>(constant->m_data);
        if (itag_buffer && itag_buffer->is_mapped()) {
            desc.m_data_mmaped = true;
        }
        return desc;
    }

    bool is_mmaped() const {
        return m_data_mmaped;
    }
private:
    ConstantDescriptor() = default;
    bool m_data_mmaped = false;
};
}  // namespace op::util
}  // namespace ov
