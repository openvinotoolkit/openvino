// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/op/constant.hpp"

namespace ov::frontend::gguf {

typedef std::map<std::string, Output<Node>> TensorMap;
struct ConstantCache {
    std::map<std::tuple<ov::element::Type_t, ov::Shape, std::vector<int64_t>>, std::shared_ptr<ov::op::v0::Constant>>
        constants;
    std::map<std::pair<ov::element::Type_t, size_t>, std::shared_ptr<ov::op::v0::Constant>> hadamard_matrices;
    std::map<std::tuple<Output<Node>,
                        int64_t,
                        bool,
                        bool,
                        std::vector<int64_t>,
                        ov::element::Type_t,
                        ov::Shape,
                        std::vector<uint8_t>>,
             Output<Node>>
        hadamard_outputs;
    std::map<Output<Node>, Output<Node>> f32_activations;
    std::map<Output<Node>, Output<Node>> rank3_activations;
};

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgufDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                const std::shared_ptr<ConstantCache>& constant_cache = {})
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_constant_cache(constant_cache ? constant_cache : std::make_shared<ConstantCache>()) {
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
            if (shape.rank().is_static() && shape.size() < 4) {
                std::vector<ov::Dimension> dims(4 - shape.size(), 1);
                dims.insert(dims.end(), shape.begin(), shape.end());
                shape = ov::PartialShape(dims);
            }
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

    int get_op_case() const {
        return get_attribute<int>("op_case", 0);
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

    std::shared_ptr<ov::op::v0::Constant> get_constant(const ov::element::Type& type,
                                                       const ov::Shape& shape,
                                                       const std::vector<int64_t>& values) const {
        auto& constant = m_constant_cache->constants[{static_cast<ov::element::Type_t>(type), shape, values}];
        if (!constant)
            constant = ov::op::v0::Constant::create(type, shape, values);
        return constant;
    }

    std::shared_ptr<ov::op::v0::Constant> get_hadamard_matrix(const ov::element::Type& type,
                                                              size_t block = 1024) const {
        const auto make_matrix = [&] {
            const float scale = 1.f / std::sqrt(static_cast<float>(block));
            std::vector<float> values(block * block);
            for (size_t row = 0; row < block; ++row) {
                for (size_t column = 0; column < block; ++column) {
                    bool negative = false;
                    for (auto bits = row & column; bits; bits &= bits - 1)
                        negative = !negative;
                    values[row * block + column] = negative ? -scale : scale;
                }
            }
            return ov::op::v0::Constant::create(type, {block, block}, values);
        };
        auto& matrix = m_constant_cache->hadamard_matrices[{static_cast<ov::element::Type_t>(type), block}];
        if (!matrix)
            matrix = make_matrix();
        return matrix;
    }

    Output<Node>* get_hadamard_cache_entry(const Output<Node>& input,
                                           int64_t block,
                                           bool inverse,
                                           bool matrix_form,
                                           const std::vector<int64_t>& heads,
                                           const ov::Tensor& signs) const;

    Output<Node> get_f32_activation(const Output<Node>& input) const;
    Output<Node> get_rank3_activation(const Output<Node>& input) const;

private:
    std::shared_ptr<GgufDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    std::shared_ptr<ConstantCache> m_constant_cache;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::gguf::NodeContext&)>;

}  // namespace ov::frontend::gguf
