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
 * @brief Sets an input/output tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor(ov_infer_request_t* infer_request, const char* tensor_name, const ov_tensor_t* tensor);

/**
 * @brief Sets an input tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the input tensor. If @p idx is greater than the number of model inputs, an exception is thrown.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, size_t idx, const ov_tensor_t* tensor);

/**
 * @brief Gets an input/output tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor(const ov_infer_request_t* infer_request, const char* tensor_name, ov_tensor_t** tensor);

/**
 * @brief Gets an output tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_output_tensor(const ov_infer_request_t* infer_request, size_t idx, ov_tensor_t** tensor);

/**
 * @brief Infers specified input(s) in synchronous mode.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_infer(ov_infer_request_t* infer_request);

/**
 * @brief Cancels inference request.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_cancel(ov_infer_request_t* infer_request);

/**
 * @brief Starts inference of specified input(s) in asynchronous mode.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_start_async(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_wait(ov_infer_request_t* infer_request);

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
