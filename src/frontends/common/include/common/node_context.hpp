// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "frontend_defs.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

class FRONTEND_API NodeContext {
public:
    NodeContext(const std::string& _op_type, OutputVector _ng_inputs) : m_op_type(_op_type), m_ng_inputs(_ng_inputs) {}
    NodeContext(const std::string& _op_type, std::map<std::string, OutputVector> _ng_inputs) : m_op_type(_op_type), m_ng_named_inputs(_ng_inputs) {}

    OutputVector get_ng_inputs() const {
        return m_ng_inputs;
    }

    std::map<std::string, OutputVector> get_ng_named_inputs() const {
        return m_ng_named_inputs;
    }

    const std::string& op_type() const {
        return m_op_type;
    }

private:
    std::string m_op_type;
    OutputVector m_ng_inputs;
    std::map<std::string, OutputVector> m_ng_named_inputs;
};

}  // namespace frontend
}  // namespace ov
