// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/model.hpp"
#include "openvino/frontend/extension.hpp"
#include "openvino/frontend/manager.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace tests {

extern const std::string TF_LITE_FE;

// A wrapper to create TensorFlow Lite Frontend and configure the conversion pipeline
std::shared_ptr<ov::Model> convert_model(const std::string& model_path,
                                         const ov::frontend::ConversionExtensionBase::Ptr& conv_ext = nullptr);
}  // namespace tests
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov

// For compatibility purposes, need to remove when will be unused
extern const std::string TF_LITE_FE;
