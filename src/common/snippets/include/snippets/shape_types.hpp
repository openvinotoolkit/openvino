// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov::snippets {
/*
 * This header file contain declarations of shape-relevant classes used cross several snippets subsystems.
 * The main purpose of storing such declarations here is to eliminate false dependencies. For example,
 * both PortDescriptor and IShapeInferSnippets use VectorDims, but these two classes are completely independent
 * semantically.
 */
using VectorDims = ov::Shape;
using VectorDimsPtr = std::shared_ptr<VectorDims>;
using VectorDimsCPtr = std::shared_ptr<const VectorDims>;
using VectorDimsRef = std::reference_wrapper<const VectorDims>;

}  // namespace ov::snippets
