// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <snippets/shape_inference/shape_inference.hpp>

#include "openvino/core/node.hpp"

namespace ov::snippets {

// Helper to register an operation with a stateless shape infer implementation
template <typename Op, typename InferType>
inline IShapeInferSnippetsFactory::TRegistry::value_type make_predefined() {
    return {Op::get_type_info_static(), []([[maybe_unused]] const std::shared_ptr<ov::Node>&) {
                return std::make_shared<InferType>();
            }};
}

// Helper to register an operation with ShapeInfer nested in the operation
template <typename Op>
inline IShapeInferSnippetsFactory::TRegistry::value_type make_specific() {
    return {Op::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) {
                return std::make_shared<typename Op::ShapeInfer>(n);
            }};
}

// Helper to register an operation with external shape infer implementation
template <typename Op, typename InferType>
inline IShapeInferSnippetsFactory::TRegistry::value_type make_specific_external() {
    return {Op::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) {
                return std::make_shared<InferType>(n);
            }};
}

}  // namespace ov::snippets
