// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "exceptions.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace paddle {

using InPortName = std::string;
using OutPortName = std::string;
using TensorName = std::string;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

class DecoderBase {
public:
    /// \brief Get attribute value by name and requested type
    ///
    /// \param name Attribute name
    /// \param type_info Attribute type information
    /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
    virtual ov::Any get_attribute(const std::string& name, const std::type_info& type_info) const = 0;

    virtual std::vector<OutPortName> get_output_names() const = 0;

    virtual size_t get_output_size() const = 0;

    /// \brief Get output port type
    ///
    /// Current API assumes that output port has only one output type.
    /// If decoder supports multiple types for specified port, it shall throw general
    /// exception
    ///
    /// \param port_name Port name for the node
    ///
    /// \return Type of specified output port
    virtual ov::element::Type get_out_port_type(const std::string& port_name) const = 0;

    virtual std::string get_op_type() const = 0;
};

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext {
    const DecoderBase& decoder;
    const NamedInputs& name_map;

public:
    NodeContext(const DecoderBase& _decoder, const NamedInputs& _name_map) : decoder(_decoder), name_map(_name_map) {}

    /// Returns node attribute by name. Returns 'def' value if attribute does not exist
    template <class T>
    T get_attribute(const std::string& name, const T& def) const {
        auto res = decoder.get_attribute(name, typeid(T));
        if (!res.empty()) {
            return res.as<T>();
        } else {
            return def;
        }
    }

    template <class T>
    T get_attribute(const std::string& name) const {
        auto res = decoder.get_attribute(name, typeid(T));
        FRONT_END_GENERAL_CHECK(!res.empty(), "Attribute with name '", name, "' does not exist");
        return res.as<T>();
    }

    template <class T>
    bool has_attribute(const std::string& name) const {
        return !decoder.get_attribute(name, typeid(T)).empty();
    }

    /// Detects if there is at least one input attached with a given name
    bool has_ng_input(const std::string& name) const {
        auto found = name_map.find(name);
        if (found != name_map.end())
            return !found->second.empty();
        return false;
    }

    /// Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    Output<Node> get_ng_input(const std::string& name) const {
        FRONT_END_GENERAL_CHECK(name_map.at(name).size() == 1);
        return name_map.at(name).at(0);
    }

    /// Returns all inputs with a given name
    OutputVector get_ng_inputs(const std::string& name) const {
        return name_map.at(name);
    }

    /// Returns all inputs in order they appear in map. This is used for FrameworkNode
    /// creation
    OutputVector get_all_ng_inputs() const {
        OutputVector res;
        for (const auto& entry : name_map) {
            res.insert(res.end(), entry.second.begin(), entry.second.end());
        }
        return res;
    }

    std::vector<OutPortName> get_output_names() const {
        return decoder.get_output_names();
    }

    ov::element::Type get_out_port_type(const std::string& port_name) const {
        return decoder.get_out_port_type(port_name);
    }

    std::string get_op_type() const {
        return decoder.get_op_type();
    }

    NamedOutputs default_single_output_mapping(const std::shared_ptr<Node>& node,
                                               const std::vector<OutPortName>& required_paddle_out_names) const;
};

inline NamedOutputs NodeContext::default_single_output_mapping(
    const std::shared_ptr<Node>& node,
    const std::vector<OutPortName>& required_paddle_out_names) const {
    NamedOutputs named_outputs;
    const auto& outputs = node->outputs();
    const auto& paddle_op_output_names = this->get_output_names();
    FRONT_END_GENERAL_CHECK(outputs.size() == 1, "OV node must have exactly one output");
    for (const auto& paddle_name : paddle_op_output_names) {
        if (std::find(required_paddle_out_names.begin(), required_paddle_out_names.end(), paddle_name) !=
            required_paddle_out_names.end())
            named_outputs[paddle_name] = {outputs[0]};
    }
    return named_outputs;
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
