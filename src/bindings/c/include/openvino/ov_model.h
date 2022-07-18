// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_model C API
 *
 * @file ov_model.h
 */

#pragma once

#include "ov_common.h"

// Model
/**
 * @defgroup Model Model
 * @ingroup openvino_c
 * Set of functions representing of Model and Node.
 * @{
 */

/**
 * @brief Release the memory allocated by ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t to free memory.
 */
OPENVINO_C_API(void) ov_model_free(ov_model_t* model);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param output_nodes A pointer to the ov_output_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_outputs(const ov_model_t* model, ov_output_node_list_t* output_nodes);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param input_nodes A pointer to the ov_input_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_inputs(const ov_model_t* model, ov_output_node_list_t* input_nodes);

/**
 * @brief Get the tensor name of ov_output_node list by index.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_any_name_by_index(ov_output_node_list_t* nodes, size_t idx, char** tensor_name);

/**
 * @brief Get the tensor name of node.
 * @ingroup Model
 * @param node A pointer to the ov_output_const_node_t.
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_any_name(ov_output_const_node_t* node, char** tensor_name);

/**
 * @brief Get the shape of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_get_shape_by_index(ov_output_node_list_t* nodes, size_t idx, ov_shape_t* shape);

/**
 * @brief Get the partial shape of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_get_partial_shape_by_index(ov_output_node_list_t* nodes, size_t idx, ov_partial_shape_t** partial_shape);

/**
 * @brief Get the tensor shape of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_element_type(ov_output_const_node_t* node, ov_element_type_e* tensor_type);
/**
 * @brief Get the tensor type of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_type tensor type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_get_element_type_by_index(ov_output_node_list_t* nodes, size_t idx, ov_element_type_e* tensor_type);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param input_node A pointer to the ov_output_const_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_const_node_t** input_node);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_node A pointer to the ov_input_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_const_node_t** input_node);

/**
 * @brief Returns true if any of the op's defined in the model contains partial shape.
 * @param model A pointer to the ov_model_t.
 */
OPENVINO_C_API(bool) ov_model_is_dynamic(const ov_model_t* model);

/**
 * @brief Do reshape in model with partial shape for a specified name.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_name(const ov_model_t* model, const char* tensor_name, const ov_partial_shape_t* partial_shape);

/**
 * @brief Do reshape in model with a list of <name, partial shape>.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param tensor_names input tensor name (char *) list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_names(const ov_model_t* model,
                          const char* tensor_names[],
                          const ov_partial_shape_t* partial_shapes[],
                          size_t cnt);

/**
 * @brief Do reshape in model with a list of <port id, partial shape>.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param ports The port list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_ports(const ov_model_t* model, size_t* ports, const ov_partial_shape_t** partial_shape, size_t cnt);

/**
 * @brief Do reshape in model for port 0.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e) ov_model_reshape(const ov_model_t* model, const ov_partial_shape_t* partial_shape);

/**
 * @brief Do reshape in model with a list of <ov_output_node_t, partial shape>.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param output_nodes The ov_output_node_t list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_nodes(const ov_model_t* model,
                          const ov_output_node_t* output_nodes[],
                          const ov_partial_shape_t* partial_shapes[],
                          size_t cnt);

/**
 * @brief Gets the friendly name for a model.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param friendly_name the model's friendly name.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name);

/**
 * @brief free ov_output_node_list_t
 * @param output_nodes The pointer to the instance of the ov_output_node_list_t to free.
 */
OPENVINO_C_API(void) ov_output_node_list_free(ov_output_node_list_t* output_nodes);

/**
 * @brief free ov_output_const_node_t
 * @ingroup Model
 * @param output_node The pointer to the instance of the ov_output_const_node_t to free.
 */
OPENVINO_C_API(void) ov_output_node_free(ov_output_const_node_t* output_node);

/** @} */  // end of Model
