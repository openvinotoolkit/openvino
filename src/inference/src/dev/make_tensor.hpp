// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"
#include "openvino/runtime/itensor.hpp"

namespace ov {

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param allocator allocates memory for internal tensor storage
 */
std::shared_ptr<ITensor> make_tensor(const element::Type type, const Shape& shape, const Allocator& allocator = {});

/**
 * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
 * @note Does not perform memory allocation internally
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
 * on shape and element size
 */
std::shared_ptr<ITensor> make_tensor(const element::Type type,
                                     const Shape& shape,
                                     void* host_ptr,
                                     const Strides& strides = {});

/**
 * @brief Constructs region of interest (ROI) tensor form another tensor.
 * @note Does not perform memory allocation internally
 * @param other original tensor
 * @param begin start coordinate of ROI object inside of the original object.
 * @param end end coordinate of ROI object inside of the original object.
 * @note A Number of dimensions in `begin` and `end` must match number of dimensions in `other.get_shape()`
 */
std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<ITensor>& other,
                                     const Coordinate& begin,
                                     const Coordinate& end);

IE_SUPPRESS_DEPRECATED_START
/** @cond INTERNAL */
std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<InferenceEngine::Blob>& tensor);
const InferenceEngine::Blob* get_hardware_blob(const InferenceEngine::Blob* blob);
InferenceEngine::Blob* get_hardware_blob(InferenceEngine::Blob* blob);

std::shared_ptr<InferenceEngine::Blob> tensor_to_blob(const std::shared_ptr<ITensor>& tensor, bool unwrap = true);
/** @endcond */

IE_SUPPRESS_DEPRECATED_END
}  // namespace ov
