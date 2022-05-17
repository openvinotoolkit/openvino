/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef ONEAPI_DNNL_DNNL_OCL_H
#define ONEAPI_DNNL_DNNL_OCL_H

#include "oneapi/dnnl/dnnl.h"

#include "oneapi/dnnl/dnnl_ocl_types.h"

/// @cond DO_NOT_DOCUMENT_THIS
// Set target version for OpenCL explicitly to suppress a compiler warning.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include <CL/cl.h>
/// @endcond

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_ocl_interop
/// @{

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl_memory_set_data_handle() has been called, if @p memory_kind is equal
///   to dnnl_ocl_interop_usm, or
/// - dnnl_ocl_interop_memory_set_mem_object() has been called, if @p memory_kind
///   is equal to dnnl_ocl_interop_buffer.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param memory_kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl_ocl_interop_usm.
///     - An OpenCL buffer. In this case the library doesn't own the buffer.
///       Requires @p memory_kind be equal to be equal to dnnl_ocl_interop_buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer that corresponds to the memory allocation kind
///       @p memory_kind for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_create(dnnl_memory_t *memory,
        const dnnl_memory_desc_t *memory_desc, dnnl_engine_t engine,
        dnnl_ocl_interop_memory_kind_t memory_kind, void *handle);

/// Returns the memory allocation kind associated with a memory object.
///
/// @param memory Memory to query.
/// @param memory_kind Output underlying memory allocation kind of the memory
///     object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_get_memory_kind(
        const_dnnl_memory_t memory,
        dnnl_ocl_interop_memory_kind_t *memory_kind);

/// Returns an OpenCL memory object associated with a memory object.
///
/// @param memory Memory object.
/// @param mem_object Output OpenCL memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_get_mem_object(
        const_dnnl_memory_t memory, cl_mem *mem_object);

/// Sets OpenCL memory object associated with a memory object.
///
/// For behavioral details, see dnnl_memory_set_data_handle().
///
/// @param memory Memory object.
/// @param mem_object OpenCL memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_set_mem_object(
        dnnl_memory_t memory, cl_mem mem_object);

/// Creates an engine associated with an OpenCL device and an OpenCL context.
///
/// @param engine Output engine.
/// @param device Underlying OpenCL device to use for the engine.
/// @param context Underlying OpenCL context to use for the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create(
        dnnl_engine_t *engine, cl_device_id device, cl_context context);

/// Returns the OpenCL context associated with an engine.
///
/// @param engine Engine to query.
/// @param context Output underlying OpenCL context of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_engine_get_context(
        dnnl_engine_t engine, cl_context *context);

/// Returns the OpenCL device associated with an engine.
///
/// @param engine Engine to query.
/// @param device Output underlying OpenCL device of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_get_device(
        dnnl_engine_t engine, cl_device_id *device);

/// Creates an execution stream for a given engine associated with
/// an OpenCL command queue.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param queue OpenCL command queue to use.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_stream_create(
        dnnl_stream_t *stream, dnnl_engine_t engine, cl_command_queue queue);

/// Returns the OpenCL command queue associated with an execution stream.
///
/// @param stream Execution stream to query.
/// @param queue Output OpenCL command queue.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_stream_get_command_queue(
        dnnl_stream_t stream, cl_command_queue *queue);

/// @} dnnl_api_ocl_interop

/// @} dnnl_api_interop

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif
