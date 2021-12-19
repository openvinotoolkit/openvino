// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tensorflow_frontend/utility.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/core/extension.hpp"
#include "tensorflow_frontend/frontend.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace tf {

class TF_API ConversionExtension : public ov::frontend::ConversionExtensionBase<OutputVector> {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;

    ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
        : ConversionExtensionBase(op_type, &converter) {}
};

}  // namespace tf
}  // namespace frontend
}  // namespace ov