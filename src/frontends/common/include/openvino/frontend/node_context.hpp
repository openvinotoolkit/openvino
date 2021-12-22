// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

template <class T>
class FRONTEND_API NodeContext {
public:
    NodeContext(const std::string& _op_type, const T& inputs) : m_op_type(_op_type), m_inputs(inputs) {}

    virtual T get_inputs() const {
        return m_inputs;
    }

    /// Get operation type
    virtual const std::string& get_op_type() const {
        return m_op_type;
    }

    template <class D>
    D get_attribute (const std::string& name) {
        return get_attribute_as_any(name).template as<D>();
    }

protected:

    virtual ov::Any get_attribute_as_any (const std::string& name) const = 0;
private:
    std::string m_op_type;
    T m_inputs;
};

template <class T>
using CreatorFunction = std::function<T(const NodeContext<T>&)>;

}  // namespace frontend
}  // namespace ov
