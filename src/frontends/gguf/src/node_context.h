// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include "openvino/frontend/gguf/decoder.h"

namespace ov {
namespace frontend {
namespace gguf {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgufDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const {
        return m_input_names;
    }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    int64_t get_input_view_offset(size_t index) const {
        return m_decoder->get_input_view_offset(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const {
        return m_output_names[0];
    }

    PartialShape get_output_shape() const {
        return m_decoder->get_output_shape(m_node_idx);
    }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(m_node_idx, name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const {
        return m_decoder->is_static();
    }

    bool is_stateful() const {
        return m_decoder->is_stateful();
    }

private:
    std::shared_ptr<GgufDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::gguf::NodeContext&)>;

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
