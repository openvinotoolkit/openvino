// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "common/conversion_extension.hpp"
#include "exceptions.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/variant.hpp"
#include "onnx_frontend/frontend.hpp"

namespace ov {
namespace frontend {
namespace onnx {
    class ConversionExtension : public ConversionExtensionBase {
    public:
        using Ptr = std::shared_ptr<ConversionExtension>;
        ConversionExtension() = delete;
        ConversionExtension(const std::string &op_type, const FrontEndONNX::CreatorFunction &converter)
                : ConversionExtensionBase(op_type, converter) {
        }

    private:
        using ConversionExtensionBase::get_converter;
    };
}
}  // namespace frontend
}  // namespace ov