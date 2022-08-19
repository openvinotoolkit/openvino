// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_model C API, which is a C wrapper for ov::Node class.
 *
 * @file ov_node.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_partial_shape.h"
#include "openvino/c/ov_shape.h"

typedef struct ov_output_const_node ov_output_const_node_t;
typedef struct ov_output_node ov_output_node_t;

/**
 * @struct ov_output_node_list_t
 * @brief Reprents an array of ov_output_nodes.
 */
typedef struct {
    ov_output_const_node_t* output_nodes;
    size_t size;
} ov_output_node_list_t;

// Node
/**
 * @defgroup node node
 * @ingroup openvino_c
 * Set of functions representing of Model and Node.
 * @{
 */

/**
 * @brief Get the shape of ov_output_node.
 * @ingroup node
 * @param nodes A pointer to ov_output_const_node_t.
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_shape(ov_output_const_node_t* node, ov_shape_t* tensor_shape);

/**
 * @brief Get the tensor name of ov_output_node list by index.
 * @ingroup node
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_any_name_by_index(const ov_output_node_list_t* nodes, size_t idx, char** tensor_name);

/**
 * @brief Get the shape of ov_output_node.
 * @ingroup node
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_shape_by_index(const ov_output_node_list_t* nodes, size_t idx, ov_shape_t* shape);

/**
 * @brief Get the partial shape of ov_output_node.
 * @ingroup node
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_partial_shape_by_index(const ov_output_node_list_t* nodes,
                                        size_t idx,
                                        ov_partial_shape_t* partial_shape);

/**
 * @brief Get the tensor type of ov_output_node.
 * @ingroup node
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_type tensor type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_element_type_by_index(const ov_output_node_list_t* nodes, size_t idx, ov_element_type_e* tensor_type);

/**
 * @brief free ov_output_node_list_t
 * @ingroup node
 * @param output_nodes The pointer to the instance of the ov_output_node_list_t to free.
 */
OPENVINO_C_API(void) ov_output_node_list_free(ov_output_node_list_t* output_nodes);

/**
 * @brief free ov_output_const_node_t
 * @ingroup node
 * @param output_node The pointer to the instance of the ov_output_const_node_t to free.
 */
OPENVINO_C_API(void) ov_output_node_free(ov_output_const_node_t* output_node);

/** @} */  // end of Node
