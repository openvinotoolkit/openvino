// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include "openvino/frontend/gguf/decoder.hpp"

namespace ov {
namespace frontend {
namespace gguf {

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgufDecoder>& decoder, std::shared_ptr<TensorMap>& tensor_map)
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_tensor_map(tensor_map) {
        m_input_names = decoder->get_input_names();
        m_output_names = decoder->get_output_names();
    }

    size_t get_input_size() const override {
        return m_decoder->get_input_size();
    }

    int64_t get_input_view_element_offset(size_t index) const {
        return m_decoder->get_input_view_element_offset(m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_input_names[input_index]);
    }

    PartialShape get_output_shape() const {
        return m_decoder->get_output_shape();
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        auto it = m_tensor_map->find(name);
        if (it == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return it->second;
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name();
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

private:
    std::shared_ptr<GgufDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::gguf::NodeContext&)>;

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
