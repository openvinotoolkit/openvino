// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
/**
 * @brief Creates a Tensor backed by the given MappedMemory object.
 * @note Can be removed after fix provided in CVS-187957
 *
 * @param mapped_memory MappedMemory backing the tensor; must not be null.
 * @param element_type  Element type of the resulting tensor.
 * @param partial_shape Shape hint; a single dynamic dimension is resolved from mapped_memory->size().
 */
OPENVINO_API Tensor read_tensor_data_mmap_impl(std::shared_ptr<MappedMemory> mapped_memory,
                                               const element::Type& element_type,
                                               const PartialShape& partial_shape);
}  // namespace ov
