// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov::frontend::gguf {

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

    const std::vector<std::string>& get_input_names() const {
        return m_input_names;
    }

    size_t get_input_size() const override {
        return m_decoder->get_input_size();
    }

    PartialShape get_input_shape(size_t input_index) const {
        auto shape = m_decoder->get_input_shape(m_input_names[input_index]);
        if (shape.rank().is_dynamic()) {
            shape = get_input(static_cast<int>(input_index)).get_partial_shape();
        }
        return shape;
    }

    // Element offset of a VIEW input into a larger tensor (0 when not a view). The decoder
    // already divides ggml's raw byte offset by element size, so translators work in elements.
    int64_t get_input_view_element_offset(size_t index) const {
        return m_decoder->get_input_view_element_offset(m_input_names[index]);
    }

    PartialShape get_output_shape() const {
        return m_decoder->get_output_shape();
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names.at(idx));
    }

    Output<Node> get_input(const std::string& name) const override {
        FRONT_END_GENERAL_CHECK(std::find(m_input_names.begin(), m_input_names.end(), name) != m_input_names.end(),
                                "Input is not an operand of this operation: ", name);
        auto it = m_tensor_map->find(name);
        if (it == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return it->second;
    }

    bool has_input(const std::string& name) const {
        return std::find(m_input_names.begin(), m_input_names.end(), name) != m_input_names.end() &&
               m_tensor_map->find(name) != m_tensor_map->end();
    }

    // Per-translation memo for values that several nodes would otherwise rebuild identically (e.g. the
    // RoPE sin/cos table shared by every layer). Entries live in the tensor map under a reserved prefix
    // so they never collide with a GGML tensor name.
    bool has_cached(const std::string& key) const {
        return m_tensor_map->find(kCachePrefix + key) != m_tensor_map->end();
    }

    Output<Node> get_cached(const std::string& key) const {
        return m_tensor_map->at(kCachePrefix + key);
    }

    void set_cached(const std::string& key, const Output<Node>& value) const {
        (*m_tensor_map)[kCachePrefix + key] = value;
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name();
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

private:
    static constexpr const char* kCachePrefix = "__gguf_cache__/";
    std::shared_ptr<GgufDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::gguf::NodeContext&)>;

}  // namespace ov::frontend::gguf
