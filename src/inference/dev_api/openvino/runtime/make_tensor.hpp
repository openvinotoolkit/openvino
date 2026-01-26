// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Helpers to create OpenVINO Tensors
 * @file openvino/runtime/make_tensor.hpp
 */

#pragma once

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/allocator.hpp"

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
 * @brief Constructs Tensor using element type and shape. Wraps allocated host memory as read-only.
 * @note Does not perform memory allocation internally
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
 * on shape and element size
 */
OPENVINO_RUNTIME_API std::shared_ptr<ITensor> make_tensor(const element::Type type,
                                                          const Shape& shape,
                                                          const void* host_ptr,
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

/**
 * @brief Returns the byte offset of the tensor's data pointer from the base memory allocation
 *
 * @details This function is useful for ROI (Region of Interest) tensors that share memory with a parent tensor
 * but start at a different position within that memory.
 *
 * @param tensor Tensor implementation to get the offset from.
 *
 * @return The offset in bytes from the base memory pointer. Returns 0 if the tensor starts at the
 *         beginning of its memory allocation or if no offset is applicable.
 */
OPENVINO_RUNTIME_API size_t get_tensor_data_offset(const ov::ITensor& tensor);

}  // namespace ov
