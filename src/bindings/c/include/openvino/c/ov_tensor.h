// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_tensor C API, which is a wrapper for ov::Tensor class
 * Tensor API holding host memory
 * @file ov_tensor.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_partial_shape.h"
#include "openvino/c/ov_shape.h"

typedef struct ov_tensor ov_tensor_t;

// Tensor
/**
 * @defgroup tensor tensor
 * @ingroup openvino_c
 * Set of functions representing of tensor.
 * @{
 */

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @ingroup tensor
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
 * @ingroup tensor
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor);

/**
 * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
 * @ingroup tensor
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape);

/**
 * @brief Get shape for tensor.
 * @ingroup tensor
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape);

/**
 * @brief Get type for tensor.
 * @ingroup tensor
 * @param type Tensor element type
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type);

/**
 * @brief the total number of elements (a product of all the dims or 1 for scalar).
 * @ingroup tensor
 * @param elements_size number of elements
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size);

/**
 * @brief the size of the current Tensor in bytes.
 * @ingroup tensor
 * @param byte_size the size of the current Tensor in bytes.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size);

/**
 * @brief Provides an access to the underlaying host memory.
 * @ingroup tensor
 * @param data A point to host memory.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_data(const ov_tensor_t* tensor, void** data);

/**
 * @brief Free ov_tensor_t.
 * @ingroup tensor
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(void) ov_tensor_free(ov_tensor_t* tensor);

/** @} */  // end of tensor
