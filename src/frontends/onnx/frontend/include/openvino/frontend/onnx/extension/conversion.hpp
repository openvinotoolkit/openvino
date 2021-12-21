// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/onnx/node_context.hpp"
#include "openvino/frontend/onnx/visibility.hpp"

namespace ov {
namespace frontend {
namespace onnx {
class ONNX_FRONTEND_API ConversionExtension : public ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;
    ConversionExtension() = delete;
    ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}
    CreatorFunction get_converter() {
        return m_converter;
    }

private:
    CreatorFunction m_converter;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov