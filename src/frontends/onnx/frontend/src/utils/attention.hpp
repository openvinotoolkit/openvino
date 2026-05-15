// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace attention {

/// \brief Extracts specific dimensions from a ShapeOf node using Gather.
///
/// \param shape  A ShapeOf node whose output contains the tensor's shape.
/// \param dims   Indices of the dimensions to extract.
/// \return A Gather node producing the selected dimension values.
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);

/// \brief Convenience overload: computes ShapeOf for the given node, then extracts dimensions.
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

}  // namespace attention
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
