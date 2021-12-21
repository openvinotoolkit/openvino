// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx_import/core/node.hpp>
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/onnx/frontend.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {
class ONNX_FRONTEND_API NodeContext : public ov::frontend::NodeContext<OutputVector> {
public:
    explicit NodeContext(const ngraph::onnx_import::Node& _context)
        : ov::frontend::NodeContext<OutputVector>(_context.op_type(), _context.get_ng_inputs()),
          context(_context) {}

protected:
    const ngraph::onnx_import::Node& context;
};
using CreatorFunction = std::function<OutputVector(const ov::frontend::onnx::NodeContext&)>;
}  // namespace onnx
}  // namespace frontend
}  // namespace ov