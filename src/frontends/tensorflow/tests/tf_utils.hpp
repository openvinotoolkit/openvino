// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/frontend/extension.hpp>
#include <string>
#include <vector>

namespace ov {
namespace frontend {
namespace tensorflow {
namespace tests {
extern const std::string TF_FE;

// a wrapper to create TensorFlow Frontend and configure the conversion pipeline
// by registering new translator via extension, specifying (new) inputs, their shapes and types
std::shared_ptr<Model> convert_model(const std::string& model_path,
                                     const ov::frontend::ConversionExtension::Ptr& conv_ext = nullptr,
                                     const std::vector<std::string>& input_names = {},
                                     const std::vector<ov::element::Type>& input_types = {},
                                     const std::vector<ov::PartialShape>& input_shapes = {},
                                     const std::vector<std::string>& input_names_to_freeze = {},
                                     const std::vector<void*>& freeze_values = {},
                                     const bool disable_mmap = false,
                                     const std::vector<std::string>& output_names = {});

}  // namespace tests
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
