// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/jax/decoder.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace jax {

class TranslateSession;

typedef std::unordered_map<size_t, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(std::shared_ptr<JaxDecoder> decoder,
                const TensorMap& ext_tensor_map,
                std::shared_ptr<TensorMap> tensor_map,
                std::shared_ptr<ParameterVector> external_parameters,
                TranslateSession* translate_session)
        : frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_ext_tensor_map(ext_tensor_map),
          m_tensor_map(tensor_map),
          m_external_parameters(external_parameters),
          m_translate_session(translate_session),
          m_decoder_inputs(decoder->inputs()),
          m_decoder_outputs(decoder->outputs()) {
        FRONT_END_GENERAL_CHECK(m_tensor_map != nullptr && m_external_parameters != nullptr &&
                                m_translate_session != nullptr);
    }

    // Do not search for input in tensor map; try to access it as a constant of
    // specified type T and return its value
    template <typename T>
    T const_input(size_t index) const;

    size_t get_input_size() const override {
        return m_decoder_inputs.size();
    };

    // Search for input in tensor map and return an output port for already
    // converted op
    // TODO: int due to base class uses it, but naturally it should be size_t for
    // PT
    Output<Node> get_input(int index) const override {
        auto input = m_decoder_inputs.at(index);
        FRONT_END_GENERAL_CHECK(m_tensor_map->count(input), "No tensor corresponding input: ", input, " exist.");
        return m_tensor_map->at(input);
    }

    Output<Node> get_input(const std::string& name) const override {
        FRONT_END_GENERAL_CHECK(has_attribute(name), "Input with name ", name, " doesn't exist");
        auto attr = get_attribute_as_any(name);
        if (attr.is<Output<Node>>()) {
            // Case when input is constant value
            return attr.as<Output<Node>>();
        } else if (attr.is<type::PyNone>()) {
            FRONT_END_THROW("Got a none input in the JAX frontend.");
        }
        FRONT_END_GENERAL_CHECK(false, "Input has type which can't be converted to ov::Node.");
    }

    Any get_values_from_const_input(int index) const override;

    // TODO: upstream to base class
    OutputVector inputs() const {
        OutputVector res;
        for (auto input : m_decoder_inputs) {
            FRONT_END_GENERAL_CHECK(m_tensor_map->count(input), "No tensor corresponding index: ", input, " exist.");
            res.push_back(m_tensor_map->at(input));
        }
        return res;
    }

    Any get_input_type(size_t index) const {
        return m_decoder->get_input_type(index);
    }

    Any get_output_type(size_t index) const {
        return m_decoder->get_output_type(index);
    }

    size_t get_output_size() const {
        return m_decoder_outputs.size();
    }

    std::vector<size_t> outputs() const {
        return m_decoder_outputs;
    }

    // Convert the resulting value of this node to ov Constant; works correctly
    // only for nodes that produce constant value, naturally for prim::Constant
    OutputVector as_constant() const;

    std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const;

    // Call mark_node for each node from the vector
    void mark_nodes(std::vector<std::shared_ptr<Node>> ov_nodes) const {
        for (auto& ov_node : ov_nodes) {
            mark_node(ov_node);
        }
    }

    // Syntactic sugar around mark_node -- just calls it for corresponding node
    // for the passed output port
    Output<Node> mark_output(Output<Node> ov_output) const {
        mark_node(ov_output.get_node_shared_ptr());
        return ov_output;
    }

    Any get_attribute_as_any(const std::string& name) const override {
        FRONT_END_THROW("Attribute is not expected to appear in JAX. Implement it if it does appear.");
    }

    std::shared_ptr<JaxDecoder> get_decoder() const {
        return m_decoder;
    }

    TranslateSession* get_session() const {
        return m_translate_session;
    }

    Output<Node> get_tensor_from_model(size_t index) const {
        if (m_tensor_map->find(index) != m_tensor_map->end()) {
            return m_tensor_map->at(index);
        } else {
            return Output<Node>();
        }
    }

private:
    std::shared_ptr<JaxDecoder> m_decoder;
    const TensorMap& m_ext_tensor_map;
    std::shared_ptr<TensorMap> m_tensor_map;
    std::shared_ptr<ParameterVector> m_external_parameters;
    TranslateSession* m_translate_session = nullptr;
    const std::vector<size_t> m_decoder_inputs;
    const std::vector<size_t> m_decoder_outputs;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::jax::NodeContext&)>;

}  // namespace jax
}  // namespace frontend
}  // namespace ov