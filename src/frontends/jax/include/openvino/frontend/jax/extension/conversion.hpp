// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/frontend/jax/visibility.hpp"

namespace ov {
namespace frontend {
namespace jax {

class ConversionExtension : public ConversionExtensionBase {
public:
    OPENVINO_RTTI("frontend::jax::ConversionExtension", "", ConversionExtensionBase);

    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunction& converter)
        : ConversionExtensionBase(op_type),
          m_converter(converter) {}

    const ov::frontend::CreatorFunction& get_converter() const {
        return m_converter;
    }

private:
    ov::frontend::CreatorFunction m_converter;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
