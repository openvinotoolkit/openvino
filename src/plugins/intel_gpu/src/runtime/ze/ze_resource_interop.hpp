// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Provide import/export functions for selected ze_resource objects.
 *
 * Wraps basic interop operations from ze_ocl_interop.hpp so that it works with ze_resource.
 * @file ze_resource_interop.hpp
 */

#pragma once

#include "ze_ocl_interop.hpp"
#include "ze_resource.hpp"

namespace cldnn::ze {
/// @brief Import Level Zero driver resource from OpenCL device.
/// @param ocl_device OpenCL device handle.
/// @return Level Zero driver resource extracted from OpenCL device handle.
ze_driver_resource ze_import_driver(cl_device_id ocl_device);

/// @brief Import Level Zero device resource from OpenCL device.
/// @param ocl_device OpenCL device handle.
/// @return Level Zero device resource extracted from OpenCL device handle.
ze_device_resource ze_import_device(cl_device_id ocl_device);

/// @brief Import Level Zero context resource from OpenCL context.
/// @param ocl_context OpenCL context handle.
/// @return Level Zero context resource extracted from OpenCL context handle.
ze_context_resource ze_import_context(cl_context ocl_context);

/// @brief Import Level Zero immediate command list resource from OpenCL command queue.
/// @param ocl_command_queue OpenCL command queue handle.
/// @return Level Zero immediate command list resource extracted from OpenCL command queue handle.
ze_command_list_resource ze_import_command_list(cl_command_queue ocl_command_queue);

/// @brief Import Level Zero USM (Unified Shared Memory) resource from OpenCL memory object.
/// @param ocl_buffer OpenCL memory object handle.
/// @param context Level Zero context resource.
/// @return Level Zero USM resource extracted from OpenCL memory object handle.
ze_usm_resource ze_import_usm(cl_mem ocl_buffer, ze_context_resource context);

/// @brief Import Level Zero image resource from OpenCL image object.
/// @param ocl_image OpenCL image object handle.
/// @return Level Zero image resource extracted from OpenCL image object handle.
ze_image_resource ze_import_image(cl_mem ocl_image);

/// @brief Exports Level Zero device to OpenCL device. Does nothing if resource already has OpenCL handle.
/// @param device Device resource to export and attach OpenCL handle to.
void ze_export_ocl_device(ze_device_resource device);

/// @brief Exports Level Zero context to OpenCL context. Does nothing if resource already has OpenCL handle.
/// @param context Context resource to export and attach OpenCL handle to.
/// @param device Device resource to use for OpenCL context creation. If device does not have OpenCL handle, it will be
/// exported first.
void ze_export_ocl_context(ze_context_resource context, ze_device_resource device);

/// @brief Exports Level Zero command list to OpenCL command queue. Does nothing if resource already has OpenCL handle.
/// @param cmd_list Command list resource to export and attach OpenCL handle to.
/// @param context Context resource to use for OpenCL command queue creation. If context does not have OpenCL handle, it
/// will be exported first.
/// @param device Device resource to use for OpenCL command queue creation. If device does not have OpenCL handle, it
/// will be exported first.
void ze_export_ocl_command_queue(ze_command_list_resource cmd_list,
                                 ze_context_resource context,
                                 ze_device_resource device);

/// @brief Exports Level Zero USM (Unified Shared Memory) to OpenCL memory object. Does nothing if resource already has
/// OpenCL handle.
/// @param usm USM resource to export and attach OpenCL handle to.
/// @param context Context resource to use for OpenCL memory object creation. If context does not have OpenCL handle, it
/// will be exported first.
/// @param device Device resource to use for OpenCL memory object creation. If device does not have OpenCL handle, it
/// will be exported first.
/// @param flags OpenCL memory flags for the memory object.
/// @param size Size of the memory object.
void ze_export_ocl_mem(ze_usm_resource usm,
                       ze_context_resource context,
                       ze_device_resource device,
                       cl_mem_flags flags,
                       size_t size);

/// @brief Exports Level Zero image to OpenCL image object. Does nothing if resource already has OpenCL handle.
/// @param image Image resource to export and attach OpenCL handle to.
/// @param context Context resource to use for OpenCL image object creation. If context does not have OpenCL handle, it
/// will be exported first.
/// @param device Device resource to use for OpenCL image object creation. If device does not have OpenCL handle, it
/// will be exported first.
/// @param flags OpenCL memory flags for the image object.
/// @param format OpenCL image format for the image object.
/// @param desc OpenCL image descriptor for the image object.
void ze_export_ocl_image(ze_image_resource image,
                         ze_context_resource context,
                         ze_device_resource device,
                         cl_mem_flags flags,
                         cl_image_format format,
                         cl_image_desc desc);

}  // namespace cldnn::ze
