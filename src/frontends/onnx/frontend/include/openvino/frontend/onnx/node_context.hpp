// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
class Node;
}
}  // namespace ngraph

namespace ov {
namespace frontend {
namespace onnx {

class ONNX_FRONTEND_API NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    explicit NodeContext(const ngraph::onnx_import::Node& context);
    size_t get_input_size() const override;

    Output<ov::Node> get_input(int port_idx) const override;

    ov::Any get_attribute_as_any(const std::string& name) const override;

protected:
    const ngraph::onnx_import::Node& m_context;
    OutputVector m_inputs;

private:
    ov::Any apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const override;
};
using CreatorFunction = std::function<OutputVector(const ngraph::onnx_import::Node&)>;
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
OPENVINO_SUPPRESS_DEPRECATED_END
