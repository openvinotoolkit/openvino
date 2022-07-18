// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_tensor C API
 *
 * @file ov_tensor.h
 */

#pragma once

#include "ov_common.h"

// Tensor
/**
 * @defgroup Tensor Tensor
 * @ingroup openvino_c
 * Set of functions representing of Tensor, Shape, PartialShape, etc.
 * @{
 */

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @ingroup Tensor
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_tensor_create_from_host_ptr(const ov_element_type_e type,
                               const ov_shape_t shape,
                               void* host_ptr,
                               ov_tensor_t** tensor);

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @ingroup Tensor
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor);

/**
 * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
 * @ingroup Tensor
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(ov_status_e) ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape);

/**
 * @brief Get shape for tensor.
 * @ingroup Tensor
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape);

/**
 * @brief Get type for tensor.
 * @ingroup Tensor
 * @param type Tensor element type
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type);

/**
 * @brief the total number of elements (a product of all the dims or 1 for scalar).
 * @ingroup Tensor
 * @param elements_size number of elements
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size);

/**
 * @brief the size of the current Tensor in bytes.
 * @ingroup Tensor
 * @param byte_size the size of the current Tensor in bytes.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size);

/**
 * @brief Provides an access to the underlaying host memory.
 * @ingroup Tensor
 * @param data A point to host memory.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_data(const ov_tensor_t* tensor, void** data);

/**
 * @brief Free ov_tensor_t.
 * @ingroup Tensor
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(void) ov_tensor_free(ov_tensor_t* tensor);

/**
 * @brief Create a rank object
 * @ingroup Tensor
 * @param min_dimension The lower inclusive limit for the dimension
 * @param max_dimension The upper inclusive limit for the dimension
 * @param ov_status_e a status code.
 */

/** @} */  // end of Tensor
