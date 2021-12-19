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

template<class T>
class FRONTEND_API NodeContext {
public:
    NodeContext(const std::string& _op_type, const T& inputs)
    : m_op_type(_op_type), m_inputs(inputs) {}

    T get_inputs() const {
        return m_inputs;
    }

    const std::string& op_type() const {
        return m_op_type;
    }

    virtual ~NodeContext() = 0;
private:
    std::string m_op_type;
    T m_inputs;
};

template<class T>
using CreatorFunction = std::function<OutputVector(const NodeContext<T>&)>;

}  // namespace frontend
}  // namespace ov
