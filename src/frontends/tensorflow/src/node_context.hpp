// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <openvino/core/any.hpp>

#include "exceptions.hpp"
#include "place.hpp"
#include "tensor.pb.h"
#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow {

using InPortName = size_t;
using OutPortName = size_t;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext {
    const DecoderBase& m_decoder;
    const NamedInputs& m_name_map;

public:
    NodeContext(const DecoderBase& decoder, const NamedInputs& name_map) : m_decoder(decoder), m_name_map(name_map) {}

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
        auto found = m_name_map.find(port_index);
        if (found != m_name_map.end())
            return !found->second.empty();
        return false;
    }

    /// Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    Output<Node> get_input(const size_t& port_index) const {
        FRONT_END_GENERAL_CHECK(m_name_map.at(port_index).size() == 1);
        return m_name_map.at(port_index).at(0);
    }

    /// Returns all inputs with a given name
    OutputVector get_inputs(const size_t& port_index) const {
        return m_name_map.at(port_index);
    }

    /// Returns all inputs in order they appear in map. This is used for FrameworkNode
    /// creation
    OutputVector get_all_inputs() const {
        OutputVector res;
        for (const auto& entry : m_name_map) {
            res.insert(res.end(), entry.second.begin(), entry.second.end());
        }
        return res;
    }

    /// Get a number of inputs
    size_t get_input_size() const {
        return m_name_map.size();
    }

    /// Get operation type
    std::string get_op_type() const {
        return m_decoder.get_op_type();
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

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
