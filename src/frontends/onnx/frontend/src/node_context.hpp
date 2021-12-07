// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/input_model.hpp>
#include <editor.hpp>
#include <fstream>
#include <onnx_import/core/node.hpp>
#include <common/node_context.hpp>

namespace ov {
namespace frontend {
namespace onnx {
class NodeContext : public ov::frontend::NodeContext {
public:
    explicit NodeContext(const ngraph::onnx_import::Node& _context)
        : ov::frontend::NodeContext(_context.op_type(), _context.get_ng_inputs()),
          context(_context) {}

protected:
    const ngraph::onnx_import::Node& context;

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return context.get_attribute_value<ov::Any>(name);
    }
};
}
}
}