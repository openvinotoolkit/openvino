// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/extension/conversion.hpp>
#include "openvino/core/extension.hpp"
#include "paddlepaddle_frontend/frontend.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
class ConversionExtensionPDPD : public ov::frontend::ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtensionPDPD>;
    ConversionExtensionPDPD() = delete;
    ConversionExtensionPDPD(const std::string& op_type, const FrontEndPDPD::CreatorFunction& converter)
        : ConversionExtensionBase(op_type, converter) {}

private:
    using ConversionExtensionBase::get_converter;
};

}
}  // namespace frontend
}  // namespace ov