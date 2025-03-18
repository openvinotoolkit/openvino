// Copyright (C) 2018-2025 Intel Corporation
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

class ConversionExtension : public ConversionExtensionBase {
public:
    OPENVINO_RTTI("frontend::tensorflow::ConversionExtension", "", ConversionExtensionBase);

    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const ov::frontend::tensorflow::CreatorFunctionIndexed& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    ConversionExtension(const std::string& op_type,
                        const ov::frontend::tensorflow::CreatorFunctionNamedAndIndexed& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    const ov::frontend::tensorflow::CreatorFunction& get_converter() const {
        return m_converter;
    }

private:
    ov::frontend::tensorflow::CreatorFunction m_converter;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
