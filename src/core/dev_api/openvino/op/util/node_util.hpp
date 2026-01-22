// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

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
}  // namespace op::util
namespace weight_sharing {
struct Extension {
public:
    static Extension get_ext(const std::shared_ptr<ov::op::v0::Constant>& constant) {
        Extension desc(constant);
        return desc;
    }

    size_t get_data_id() const {
        return m_data_id;
    }

private:
    void init_values() {
        if (auto itag_buffer = m_const->m_data->get_descriptor(); itag_buffer) {
            m_data_id = itag_buffer->get_id();
        }
    }

    Extension(const std::shared_ptr<ov::op::v0::Constant>& m_const) : m_const(m_const) {
        init_values();
    }

    const std::shared_ptr<ov::op::v0::Constant>& m_const;
    size_t m_data_id = 0;
};
}  // namespace weight_sharing
}  // namespace ov
