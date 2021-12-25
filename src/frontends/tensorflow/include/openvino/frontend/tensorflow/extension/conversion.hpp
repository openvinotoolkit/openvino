// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
class TENSORFLOW_API ConversionExtension : public ConversionExtensionBase {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;
    ConversionExtension(const std::string& op_type, const CreatorFunction& converter)
        : ov::frontend::ConversionExtensionBase(op_type),
          m_converter(converter) {}

    const CreatorFunction& get_converter() {
        return m_converter;
    }

private:
    CreatorFunction m_converter;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov