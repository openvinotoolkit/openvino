// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/frontend/exception.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

class FRONTEND_API NodeContext {
public:
    explicit NodeContext(const std::string& op_type) : m_op_type(op_type) {}

    virtual size_t get_input_size() const = 0;
    virtual size_t get_input_size(const std::string& port_name) const = 0;

    virtual Output<Node> get_input(int idx) const = 0;
    virtual Output<Node> get_input(const std::string& name, int idx) const = 0;
    virtual Output<Node> get_input(const std::string& name) const = 0;

    virtual const std::string& get_op_type() const { return m_op_type; }

    /// Returns node attribute by name
    template <class T>
    T get_attribute (const std::string& name) const {
        auto res = get_attribute_as_any(name);
        FRONT_END_GENERAL_CHECK(!res.empty(), "Attribute with name '", name, "' does not exist");
        return res.as<T>();
    }

    /// Returns node attribute by name. Returns 'def' value if attribute does not exist
    template <class T>
    T get_attribute (const std::string& name, const T& def) const {
        auto res = get_attribute_as_any(name);
        if (!res.empty()) {
            return res.as<T>();
        }
        return def;
    }

    /// Check if an attribute of a given name exists
    template <typename T>
    bool has_attribute(const std::string& name) const {
        return get_attribute_as_any(name).empty();
    }

    virtual ov::Any get_attribute_as_any (const std::string& name) const = 0;
private:
    std::string m_op_type;
};

using CreatorFunction = std::function<OutputVector(const NodeContext&)>;
using CreatorFunctionNamed = std::function<std::map<std::string, OutputVector>(const NodeContext&)>;

}  // namespace frontend
}  // namespace ov
