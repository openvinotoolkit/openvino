// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

class FRONTEND_API NodeContext {
public:
    explicit NodeContext(const std::string& op_type) : m_op_type(op_type) {}
    virtual ~NodeContext() = default;

    /// \brief  Returns a number of inputs
    virtual size_t get_input_size() const {
        FRONT_END_NOT_IMPLEMENTED(get_input_size);
    };

    /// \brief Returns a number of inputs
    virtual size_t get_input_size(const std::string& port_name) const {
        FRONT_END_NOT_IMPLEMENTED(get_input_size);
    }

    /// \brief Returns exactly one input with a given idx; throws if there is no inputs or
    /// there are more than one input
    virtual Output<Node> get_input(int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_input);
    }

    /// \brief Returns exactly one input with a given name and idx; throws if there is no inputs or
    /// there are more than one input
    virtual Output<Node> get_input(const std::string& name, int idx) const {
        FRONT_END_NOT_IMPLEMENTED(get_input);
    }

    /// \brief Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    virtual Output<Node> get_input(const std::string& name) const {
        FRONT_END_NOT_IMPLEMENTED(get_input);
    }

    virtual const std::string& get_op_type() const {
        return m_op_type;
    }

    /// \brief Returns node attribute by name.
    template <class T>
    T get_attribute(const std::string& name) const {
        auto any = get_attribute_as_any(name);
        FRONT_END_GENERAL_CHECK(!any.empty(), "Attribute with name '", name, "' does not exist");

        // sometimes we can't unambiguously recognize types in protobuf, e.g.
        // int we can interpret as int or as enum inherited from int, so
        // we have to apply additional rules based on the type (T) passed from the user.
        auto res = apply_additional_conversion_rules(any, typeid(T));
        return res.as<T>();
    }

    /// \brief Returns node attribute by name. Returns 'def' value if attribute does not exist
    template <class T>
    T get_attribute(const std::string& name, const T& def) const {
        auto any = get_attribute_as_any(name);

        // sometimes we can't unambiguously recognize types in protobuf, e.g.
        // int we can interpret as int or as enum inherited from int, so
        // we have to apply additional rules based on the type (T) passed from the user.
        auto res = apply_additional_conversion_rules(any, typeid(T));
        if (!res.empty()) {
            return res.as<T>();
        }
        return def;
    }

    /// \brief Check if an attribute of a given name exist
    bool has_attribute(const std::string& name) const {
        return !get_attribute_as_any(name).empty();
    }

    /// \brief Returns node attribute by name as ov::Any.
    virtual ov::Any get_attribute_as_any(const std::string& name) const = 0;

private:
    virtual ov::Any apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const {
        return data;
    }
    std::string m_op_type;
};

using CreatorFunction = std::function<OutputVector(const NodeContext&)>;
using CreatorFunctionNamed = std::function<std::map<std::string, OutputVector>(const NodeContext&)>;

}  // namespace frontend
}  // namespace ov
