// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace ov::snippets {
/*
 * This header file contain declarations of shape-relevant classes used cross several snippets subsystems.
 * The main purpose of storing such declarations here is to eliminate false dependencies. For example,
 * both PortDescriptor and IShapeInferSnippets use VectorDims, but these two classes are completely independent
 * semantically.
 */
using VectorDims = std::vector<size_t>;
using VectorDimsPtr = std::shared_ptr<VectorDims>;
using VectorDimsCPtr = std::shared_ptr<const VectorDims>;
using VectorDimsRef = std::reference_wrapper<const VectorDims>;

}  // namespace ov::snippets
