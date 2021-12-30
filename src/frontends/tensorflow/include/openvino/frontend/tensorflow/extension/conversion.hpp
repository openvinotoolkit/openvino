// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class TENSORFLOW_API ConversionExtension : public ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    const ov::frontend::CreatorFunction& get_converter() {
        return m_converter;
    }

    ConversionExtension(const std::string& op_type,
                        const ov::frontend::ConversionExtension::PyCreatorFunction& py_converter)
        : ConversionExtensionBase(op_type),
          m_py_converter(py_converter) {
        m_converter = [&](const ov::frontend::NodeContext& node) -> OutputVector {
            return m_py_converter(static_cast<const ov::frontend::NodeContext*>(&node));
        };
    }

private:
    ov::frontend::ConversionExtension::PyCreatorFunction m_py_converter;
    ov::frontend::CreatorFunction m_converter;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov