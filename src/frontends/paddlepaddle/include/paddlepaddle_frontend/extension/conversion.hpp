// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "paddlepaddle_frontend/utility.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/core/extension.hpp"
#include "paddlepaddle_frontend/frontend.hpp"
#include "paddlepaddle_frontend/node_context.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace pdpd {

class PDPD_API ConversionExtension : public ov::frontend::ConversionExtensionBase {
    public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
            : ConversionExtensionBase(op_type), m_converter(converter) {}

    CreatorFunction get_converter() { return m_converter; }
    private:
    CreatorFunction m_converter;
};

}
}  // namespace frontend
}  // namespace ov