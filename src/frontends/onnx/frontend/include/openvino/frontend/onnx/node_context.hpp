// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class ONNX_FRONTEND_API NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    explicit NodeContext(const Node& context);
    size_t get_input_size() const override;

    Output<ov::Node> get_input(int port_idx) const override;

    ov::Any get_attribute_as_any(const std::string& name) const override;

protected:
    const Node& m_context;
    OutputVector m_inputs;
};
using CreatorFunction = std::function<OutputVector(const ov::frontend::onnx::NodeContext&)>;
}  // namespace onnx
}  // namespace frontend
}  // namespace ov