// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/input_model.hpp>
#include <common/node_context.hpp>
#include <editor.hpp>
#include <fstream>
#include <onnx_import/core/node.hpp>
#include <openvino/core/any.hpp>

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
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov