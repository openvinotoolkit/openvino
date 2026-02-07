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


class ConstantDescriptor {
public:
    static ConstantDescriptor& get_desc(const std::shared_ptr<ov::op::v0::Constant>& constant) {
        std::lock_guard<std::mutex> lock(map_mutex);

        auto const_ptr = constant.get();
        auto it = descriptor_map.find(const_ptr);
        if (it != descriptor_map.end()) {
            return it->second;
        }
        ConstantDescriptor desc;
        init_values(desc, constant);
        auto result = descriptor_map.emplace(const_ptr, std::move(desc));
        return result.first->second;
    }

    bool is_mmaped() const {
        return m_data_mmaped;
    }

    void set_id(int id) {
        m_id = id;
    }

    int get_id() const {
        return m_id;
    }

    size_t get_data_id() const {
        return m_data_id;
    }

    std::string_view get_data_tag() const {
        return m_data_tag;
    }

private:
    static void init_values(ConstantDescriptor& desc, const std::shared_ptr<ov::op::v0::Constant>& constant) {
        if (auto itag_buffer = ov::as_itag_buffer(constant->m_data); itag_buffer) {
            desc.m_data_mmaped = itag_buffer->is_mapped();
            desc.m_data_id = itag_buffer->get_id();
            desc.m_data_tag = itag_buffer->get_tag();
        }
    }


    static std::unordered_map<ov::op::v0::Constant*, ConstantDescriptor> descriptor_map;
    static std::mutex map_mutex;

    ConstantDescriptor() = default;
    bool m_data_mmaped = false;
    size_t m_data_id = 0;
    std::string_view m_data_tag;
    int m_id = -1;
};
}  // namespace op::util
}  // namespace ov
