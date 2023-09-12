// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param allocator allocates memory for internal tensor storage
 */
OPENVINO_RUNTIME_API std::shared_ptr<ITensor> make_tensor(const element::Type type,
                                                          const Shape& shape,
                                                          const Allocator& allocator = {});

/**
 * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
 * @note Does not perform memory allocation internally
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
 * on shape and element size
 */
OPENVINO_RUNTIME_API std::shared_ptr<ITensor> make_tensor(const element::Type type,
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
OPENVINO_RUNTIME_API std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<ITensor>& other,
                                                          const Coordinate& begin,
                                                          const Coordinate& end);

/**
 * @brief Constructs public ov::Tensor class
 *
 * @param tensor Tensor implementation
 *
 * @return OpenVINO Tensor
 */
OPENVINO_RUNTIME_API ov::Tensor make_tensor(const ov::SoPtr<ITensor>& tensor);

/**
 * @brief Returns tensor implementation
 *
 * @param tensor OpenVINO Tensor
 *
 * @return SoPtr to ITensor
 */
OPENVINO_RUNTIME_API ov::SoPtr<ov::ITensor> get_tensor_impl(const ov::Tensor& tensor);

IE_SUPPRESS_DEPRECATED_START
/** @cond INTERNAL */
ov::SoPtr<ITensor> make_tensor(const std::shared_ptr<InferenceEngine::Blob>& tensor, bool unwrap = false);
const InferenceEngine::Blob* get_hardware_blob(const InferenceEngine::Blob* blob);
InferenceEngine::Blob* get_hardware_blob(InferenceEngine::Blob* blob);

OPENVINO_RUNTIME_API std::shared_ptr<InferenceEngine::Blob> tensor_to_blob(const ov::SoPtr<ITensor>& tensor,
                                                                           bool unwrap = true,
                                                                           InferenceEngine::TensorDesc desc = {});
/** @endcond */

IE_SUPPRESS_DEPRECATED_END
}  // namespace ov
