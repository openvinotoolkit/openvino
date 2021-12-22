// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <openvino/core/any.hpp>

#include "decoder.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

using InPortName = size_t;
using OutPortName = size_t;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext : public ov::frontend::NodeContext<OutputVector> {
    const DecoderBase& m_decoder;
    const OutputVector& m_inputs;

public:
    NodeContext(const DecoderBase& decoder, const OutputVector& inputs)
        : ov::frontend::NodeContext<OutputVector>(decoder.get_op_type(), inputs),
          m_decoder(decoder),
          m_inputs(inputs) {}

    /// Returns node attribute by name. Returns 'def' value if attribute does not exist
    template <typename T>
    T get_attribute(const std::string& name, const T& def) const {
        auto res = m_decoder.get_attribute(name, typeid(T));
        if (!res.empty()) {
            return res.as<T>();
        }
        return def;
    }

    /// Returns node attribute by name
    template <typename T>
    T get_attribute(const std::string& name) const {
        auto res = m_decoder.get_attribute(name, typeid(T));
        FRONT_END_GENERAL_CHECK(!res.empty(), "Attribute with name '", name, "' does not exist");
        return res.as<T>();
    }

    /// Check if an attribute of a given name exists
    template <typename T>
    bool has_attribute(const std::string& name) const {
        return !m_decoder.get_attribute(name, typeid(T)).empty();
    }

    /// Detects if there is at least one input attached with a given name
    bool has_input(const size_t& port_index) const {
        return m_inputs.size() >= port_index + 1;
    }

    /// Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    Output<Node> get_input(const size_t& port_index) const {
        return m_inputs.at(port_index);
    }

    /// Returns all inputs with a given name
    OutputVector get_inputs(const size_t& port_index) const {
        return {m_inputs.at(port_index)};
    }

    /// Returns all inputs in order they appear in map. This is used for FrameworkNode
    /// creation
    OutputVector get_all_inputs() const {
        return m_inputs;
    }

    /// Get a number of inputs
    size_t get_input_size() const {
        return m_inputs.size();
    }

    /// Get a node name
    std::string get_name() const {
        return m_decoder.get_op_name();
    }

    /// Get a decoder
    const DecoderBase* get_decoder() const {
        return &m_decoder;
    }
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::tensorflow::NodeContext&)>;
using TranslatorDictionaryType = std::map<const std::string, const CreatorFunction>;

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
