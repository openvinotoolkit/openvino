// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

class TENSORFLOW_LITE_API ConversionExtension : public ov::frontend::tensorflow::ConversionExtension {
public:
    using Ptr = std::shared_ptr<ConversionExtension>;

    ConversionExtension() = delete;
    ConversionExtension(const std::string& op_type, const ov::frontend::CreatorFunction& converter)
        : ov::frontend::tensorflow::ConversionExtension(op_type, converter) {}
    ~ConversionExtension() override;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov