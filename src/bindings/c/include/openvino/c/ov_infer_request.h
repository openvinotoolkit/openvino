// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ov_infer_request C API, which is a C wrapper for ov::InferRequest class
 * This is a class of infer request that can be run in asynchronous or synchronous manners.
 * @file ov_infer_request.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_tensor.h"
#include "openvino/c/ov_node.h"

typedef struct ov_infer_request ov_infer_request_t;

/**
 * @struct ov_callback_t
 * @brief Completion callback definition about the function and args
 */
typedef struct {
    void(OPENVINO_C_API_CALLBACK* callback_func)(void* args);
    void* args;
} ov_callback_t;

/**
 * @struct ov_ProfilingInfo_t
 * @brief Store profiling info data
 */
typedef struct {
    enum Status {       //!< Defines the general status of a node.
        NOT_RUN,        //!< A node is not executed.
        OPTIMIZED_OUT,  //!< A node is optimized out during graph optimization phase.
        EXECUTED        //!< A node is executed.
    } status;
    int64_t real_time;      //!< The absolute time, in microseconds, that the node ran (in total).
    int64_t cpu_time;       //!< The net host CPU time that the node ran.
    const char* node_name;  //!< Name of a node.
    const char* exec_type;  //!< Execution type of a unit.
    const char* node_type;  //!< Node type.
} ov_profiling_info_t;

/**
 * @struct ov_profiling_info_list_t
 * @brief A list of profiling info data
 */
typedef struct {
    ov_profiling_info_t* profiling_infos;
    size_t size;
} ov_profiling_info_list_t;

// infer_request
/**
 * @defgroup infer_request infer_request
 * @ingroup openvino_c
 * Set of functions representing of infer_request.
 * @{
 */

/**
 * @brief Sets an input/output tensor to infer on by the name of tensor.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor(ov_infer_request_t* infer_request, const char* tensor_name, const ov_tensor_t* tensor);

/**
 * @brief Sets an input/output tensor to infer on by the port of input/output tensor.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Port of the input or output tensor, which can be got by call interface from ov_model_t/ov_compiled_model_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor_by_port(ov_infer_request_t* infer_request, const ov_output_node_t* port, const ov_tensor_t* tensor);

/**
 * @brief Sets an input/output tensor to infer on by the const port of input/output tensor.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Const port of the input or output tensor, which can be got by call interface from ov_model_t/ov_compiled_model_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor_by_const_port(ov_infer_request_t* infer_request, const ov_output_const_node_t* port, const ov_tensor_t* tensor);

/**
 * @brief Sets a batch of tensors for input data to infer by tensor name.
 * Model input must have batch dimension, and the number of @p tensors must match the batch size.
 * The current version supports setting tensors to model inputs only. If @p tensor_name is associated
 * with output (or any other non-input node), an exception is thrown.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name Name of the input tensor.
 * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
 * input element type and shape (except batch dimension). Total size of tensors must match the input size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensors(ov_infer_request_t* infer_request, const char* tensor_name, const ov_tensor_list_t* tensors);

/**
 * @brief Sets a batch of tensors for input data to infer by input port.
 * Model input must have batch dimension, and the number of @p tensors must match the batch size.
 * The current version supports setting tensors to model inputs only. If @p port is associated
 * with output (or any other non-input node), an exception is thrown.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Const port of the input tensor.
 * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
 * input element type and shape (except batch dimension). Total size of tensors must match the input size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensors_by_const_port(ov_infer_request_t* infer_request, const ov_output_const_node_t* port, const ov_tensor_list_t* tensors);

/**
 * @brief Sets an input tensor to infer on by the index of tensor.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the input port. If @p idx is greater than the number of model inputs, an exception is thrown.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensor_by_index(ov_infer_request_t* infer_request, size_t idx, const ov_tensor_t* tensor);

/**
 * @brief Sets an input tensor for the model with single input to infer on.
 * @note If model has several inputs, an exception is thrown.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor);

/**
 * @brief Sets a batch of tensors for the model with single input to infer on.
 * Model input must have batch dimension, and the number of @p tensors must match the batch size.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
 * input element type and shape (except batch dimension). Total size of tensors must match the input size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensors(ov_infer_request_t* infer_request, const ov_tensor_list_t* tensors);

/**
 * @brief Sets a batch of tensors for input data by index.
 * Model input must have batch dimension, and the number of @p tensors must match the batch size.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the input tensor. If @p idx is greater than the number of model inputs, an exception is thrown.
 * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
 * input element type and shape (except batch dimension). Total size of tensors must match the input size.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensors_by_index(ov_infer_request_t* infer_request, size_t idx, const ov_tensor_list_t* tensors);

/**
 * @brief Sets an output tensor to infer by the index of output tensor.
 * @note Index of the output preserved accross ov_model_t, ov_compiled_model_t.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_output_tensor_by_index(ov_infer_request_t* infer_request, size_t idx, const ov_tensor_t* tensor);

/**
 * @brief Sets an output tensor to infer models with single output.
 * @note If model has several outputs, an exception is thrown.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_output_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor);

/**
 * @brief Gets an input/output tensor by the name of tensor.
 * @note If model has several outputs, an exception is thrown.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name Name of the input or output tensor to get.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor(const ov_infer_request_t* infer_request, const char* tensor_name, ov_tensor_t** tensor);

/**
 * @brief Gets an input/output tensor by const port.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Port of the tensor to get. @p port is not found, an exception is thrown.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor_by_const_port(const ov_infer_request_t* infer_request, const ov_output_const_node_t* port, ov_tensor_t** tensor);

/**
 * @brief Gets an input/output tensor by port.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param port Port of the tensor to get. @p port is not found, an exception is thrown.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor_by_port(const ov_infer_request_t* infer_request, const ov_output_node_t* port, ov_tensor_t** tensor);

/**
 * @brief Gets an input tensor by the index of input tensor.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get. @p idx. If the tensor with the specified @p idx is not found, an exception
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_input_tensor_by_index(const ov_infer_request_t* infer_request, size_t idx, ov_tensor_t** tensor);

/**
 * @brief Gets an input tensor from the model with only one input tensor.
 * @ingroup infer_request
 * @note If model has several inputs, an exception is thrown.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_input_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor);

/**
 * @brief Gets an output tensor by the index of output tensor.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get. @p idx. If the tensor with the specified @p idx is not found, an exception
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_output_tensor_by_index(const ov_infer_request_t* infer_request, size_t idx, ov_tensor_t** tensor);

/**
 * @brief Gets an output tensor from the model with only one output tensor.
 * @note f model has several outputs, an exception is thrown.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_output_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor);

/**
 * @brief Infers specified input(s) in synchronous mode.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_infer(ov_infer_request_t* infer_request);

/**
 * @brief Cancels inference request.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_cancel(ov_infer_request_t* infer_request);

/**
 * @brief Starts inference of specified input(s) in asynchronous mode.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_start_async(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_wait(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the specified timeout has elapsed or the result
 * becomes available, whichever comes first.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param timeout Maximum duration, in milliseconds, to block for.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_wait_for(ov_infer_request_t* infer_request, const int64_t timeout);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param callback  A function to be called.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback);

/**
 * @brief Release the memory allocated by ov_infer_request_t.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t to free memory.
 */
OPENVINO_C_API(void) ov_infer_request_free(ov_infer_request_t* infer_request);

/**
 * @brief Queries performance measures per layer to identify the most time consuming operation.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param profiling_infos  Vector of profiling information for operations in a model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_profiling_info(ov_infer_request_t* infer_request, ov_profiling_info_list_t* profiling_infos);

/**
 * @brief Release the memory allocated by ov_profiling_info_list_t.
 * @ingroup infer_request
 * @param profiling_infos A pointer to the ov_profiling_info_list_t to free memory.
 */
OPENVINO_C_API(void) ov_profiling_info_list_free(ov_profiling_info_list_t* profiling_infos);

/** @} */  // end of infer_request
