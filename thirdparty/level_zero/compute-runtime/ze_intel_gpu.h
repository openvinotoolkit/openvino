// intel/compute-runtime 4df478c5139703c82e548a65eafbcc69923953ac
/*
 * Copyright (C) 2020-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZE_INTEL_GPU_H
#define _ZE_INTEL_GPU_H

#include <ze_api.h>

#include "ze_stypes.h"

#if defined(__cplusplus)
#pragma once
extern "C" {
#endif

#include <stdint.h>

#define ZE_INTEL_GPU_VERSION_MAJOR 0
#define ZE_INTEL_GPU_VERSION_MINOR 1

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME
/// @brief Module DP properties driver extension name
#define ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME "ZE_intel_experimental_device_module_dp_properties"
#endif // ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Module DP properties driver extension Version(s)
typedef enum _ze_intel_device_module_dp_properties_exp_version_t {
    ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_DEVICE_MODULE_DP_PROPERTIES_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_module_dp_properties_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported Dot Product flags
typedef uint32_t ze_intel_device_module_dp_exp_flags_t;
typedef enum _ze_intel_device_module_dp_exp_flag_t {
    ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DP4A = ZE_BIT(0), ///< Supports DP4A operation
    ZE_INTEL_DEVICE_MODULE_EXP_FLAG_DPAS = ZE_BIT(1), ///< Supports DPAS operation
    ZE_INTEL_DEVICE_MODULE_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_module_dp_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device Module dot product properties queried using
///        ::zeDeviceGetModuleProperties
///
/// @details
///     - This structure may be passed to ::zeDeviceGetModuleProperties, via
///       `pNext` member of ::ze_device_module_properties_t.
/// @brief Device module dot product properties
typedef struct _ze_intel_device_module_dp_exp_properties_t {
    ze_structure_type_ext_t stype = ZE_STRUCTURE_INTEL_DEVICE_MODULE_DP_EXP_PROPERTIES; ///< [in] type of this structure
    void *pNext;                                                                        ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                                                        ///< structure (i.e. contains sType and pNext).
    ze_intel_device_module_dp_exp_flags_t flags;                                        ///< [out] 0 (none) or a valid combination of ::ze_intel_device_module_dp_flag_t
} ze_intel_device_module_dp_exp_properties_t;

#ifndef ZE_INTEL_COMMAND_LIST_MEMORY_SYNC
/// @brief Cmd List memory sync extension name
#define ZE_INTEL_COMMAND_LIST_MEMORY_SYNC "ZE_intel_experimental_command_list_memory_sync"
#endif // ZE_INTEL_COMMAND_LIST_MEMORY_SYNC

///////////////////////////////////////////////////////////////////////////////
/// @brief Cmd List memory sync extension Version(s)
typedef enum _ze_intel_command_list_memory_sync_exp_version_t {
    ZE_INTEL_COMMAND_LIST_MEMORY_SYNC_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_COMMAND_LIST_MEMORY_SYNC_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_COMMAND_LIST_MEMORY_SYNC_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_intel_command_list_memory_sync_exp_version_t;

#ifndef ZE_INTEL_STRUCTURE_TYPE_DEVICE_COMMAND_LIST_WAIT_ON_MEMORY_DATA_SIZE_EXP_DESC
/// @brief stype for _ze_intel_device_command_list_wait_on_memory_data_size_exp_desc_t
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Extended descriptor for cmd list memory sync
///
/// @details
///     - Implementation must support ::ZE_intel_experimental_command_list_memory_sync extension
///     - May be passed to ze_device_properties_t through pNext.
typedef struct _ze_intel_device_command_list_wait_on_memory_data_size_exp_desc_t {
    ze_structure_type_ext_t stype;               ///< [in] type of this structure
    const void *pNext;                           ///< [in][optional] must be null or a pointer to an extension-specific
                                                 ///< structure (i.e. contains stype and pNext).
    uint32_t cmdListWaitOnMemoryDataSizeInBytes; /// <out> Defines supported data size for zexCommandListAppendWaitOnMemory[64] API
} ze_intel_device_command_list_wait_on_memory_data_size_exp_desc_t;

#ifndef ZEX_INTEL_EVENT_SYNC_MODE_EXP_NAME
/// @brief Event sync mode extension name
#define ZEX_INTEL_EVENT_SYNC_MODE_EXP_NAME "ZEX_intel_experimental_event_sync_mode"
#endif // ZE_INTEL_EVENT_SYNC_MODE_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Event sync mode extension Version(s)
typedef enum _zex_intel_event_sync_mode_exp_version_t {
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} zex_intel_event_sync_mode_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported event sync mode flags
typedef uint32_t zex_intel_event_sync_mode_exp_flags_t;
typedef enum _zex_intel_event_sync_mode_exp_flag_t {
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_LOW_POWER_WAIT = ZE_BIT(0),          ///< Low power host synchronization mode, for better CPU utilization
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_SIGNAL_INTERRUPT = ZE_BIT(1),        ///< Generate interrupt when Event is signalled on Device
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_EXTERNAL_INTERRUPT_WAIT = ZE_BIT(2), ///< Host synchronization APIs wait for external interrupt. Can be used only for Events created via zexCounterBasedEventCreate
    ZEX_INTEL_EVENT_SYNC_MODE_EXP_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} zex_intel_event_sync_mode_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Extended descriptor for event sync mode
///
/// @details
///     - Implementation must support ::ZEX_intel_experimental_event_sync_mode extension
///     - May be passed to ze_event_desc_t through pNext.
typedef struct _zex_intel_event_sync_mode_exp_desc_t {
    ze_structure_type_ext_t stype;                       ///< [in] type of this structure
    const void *pNext;                                   ///< [in][optional] must be null or a pointer to an extension-specific
                                                         ///< structure (i.e. contains stype and pNext).
    zex_intel_event_sync_mode_exp_flags_t syncModeFlags; /// <in> valid combination of ::ze_intel_event_sync_mode_exp_flag_t
    uint32_t externalInterruptId;                        /// <in> External interrupt id. Used only when ZEX_INTEL_EVENT_SYNC_MODE_EXP_FLAG_EXTERNAL_INTERRUPT_WAIT flag is set
} zex_intel_event_sync_mode_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare zex_intel_queue_allocate_msix_hint_exp_desc_t
typedef struct _zex_intel_queue_allocate_msix_hint_exp_desc_t zex_intel_queue_allocate_msix_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue descriptor for allocating unique msix. This structure may be
/// passed as pNext member of ::ze_command_queue_desc_t.

typedef struct _zex_intel_queue_allocate_msix_hint_exp_desc_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    const void *pNext;             ///< [in][optional] must be null or a pointer to an extension-specific
                                   ///< structure (i.e. contains stype and pNext).
    ze_bool_t uniqueMsix;          ///< [in] If set, try to allocate unique msix for command queue.
                                   ///< If not set, driver will follow default behaviour. It may share msix for signaling completion with other queues.
                                   ///< Number of unique msixes may be limited. On unsuccessful allocation, queue or immediate cmd list creation API fallbacks to default behaviour.

} zex_intel_queue_allocate_msix_hint_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue descriptor for enabling copy operations offload. This structure may be
/// passed as pNext member of ::ze_command_queue_desc_t.

typedef struct _zex_intel_queue_copy_operations_offload_hint_exp_desc_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    const void *pNext;             ///< [in][optional] must be null or a pointer to an extension-specific
                                   ///< structure (i.e. contains stype and pNext).
    ze_bool_t copyOffloadEnabled;  ///< [in] If set, try to offload copy operations to different engines. Applicable only for compute queues.
                                   ///< This is only a hint. Driver may ignore it per append call, based on platform capabilities or internal heuristics.
                                   ///< If not set, driver will follow default behaviour. Copy operations will be submitted to same engine as compute operations.

} zex_intel_queue_copy_operations_offload_hint_exp_desc_t;

#ifndef ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME
/// @brief Queue copy operations offload hint extension name
#define ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME "ZEX_intel_experimental_queue_copy_operations_offload_hint"
#endif // ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Queue copy operations offload hint extension version(s)
typedef enum _zex_intel_queue_copy_operations_offload_hint_exp_version_t {
    ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZEX_INTEL_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} zex_intel_queue_copy_operations_offload_hint_exp_version_t;

#if ZE_API_VERSION_CURRENT_M <= ZE_MAKE_VERSION(1, 13)

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue flag for enabling copy operations offload
///
/// If set, try to offload copy operations to different engines. Applicable only for compute queues.
/// This is only a hint. Driver may ignore it per append call, based on platform capabilities or internal heuristics.
#define ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT ZE_BIT(2)

#endif // ZE_API_VERSION_CURRENT_M <= ZE_MAKE_VERSION(1, 13)

#ifndef ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_NAME
/// @brief Extension name for query to read the Intel Level Zero Driver Version String
#define ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_NAME "ZE_intel_get_driver_version_string"
#endif // ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Query to read the Intel Level Zero Driver Version String extension version(s)
typedef enum _ze_intel_get_driver_version_string_exp_version_t {
    ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_GET_DRIVER_VERSION_STRING_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_intel_get_driver_version_string_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported 2D Block Array flags
typedef uint32_t ze_intel_device_block_array_exp_flags_t;
typedef enum _ze_intel_device_block_array_exp_flag_t {
    ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_STORE = ZE_BIT(0), ///< Supports store operation
    ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_LOAD = ZE_BIT(1),  ///< Supports load operation
    ZE_INTEL_DEVICE_EXP_FLAG_2D_BLOCK_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_block_array_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME
/// @brief Device 2D block array properties driver extension name
#define ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME "ZE_intel_experimental_device_block_array_properties"
#endif // ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_NAME

/// @brief Device 2D block array properties queried using
///        ::zeDeviceGetProperties
///
/// @details
///     - This structure may be passed to ::zeDeviceGetProperties, via
///       `pNext` member of ::ze_device_properties_t.
/// @brief Device 2D block array properties

typedef struct _ze_intel_device_block_array_exp_properties_t {
    ze_structure_type_ext_t stype;                 ///< [in] type of this structure
    void *pNext;                                   ///< [in,out][optional] must be null or a pointer to an extension-specific
                                                   ///< structure (i.e. contains sType and pNext).
    ze_intel_device_block_array_exp_flags_t flags; ///< [out] 0 (none) or a valid combination of ::ze_intel_device_block_array_exp_flag_t
} ze_intel_device_block_array_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device 2D block array properties driver extension versions
typedef enum _ze_intel_device_block_array_exp_properties_version_t {
    ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0), ///< version 1.0
    ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_DEVICE_BLOCK_ARRAY_EXP_PROPERTIES_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_intel_device_block_array_exp_properties_version_t;

/// @brief Query to read the Intel Level Zero Driver Version String
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - The Driver Version String will be in the format:
///     - Major.Minor.Patch+Optional per semver guidelines https://semver.org/#spec-item-10
/// @returns
///     - ::ZE_RESULT_SUCCESS
ZE_APIEXPORT ze_result_t ZE_APICALL
zeIntelGetDriverVersionString(
    ze_driver_handle_t hDriver, ///< [in] Driver handle whose version is being read.
    char *pDriverVersion,       ///< [in,out] pointer to driver version string.
    size_t *pVersionSize);      ///< [in,out] pointer to the size of the driver version string.
                                ///< if size is zero, then the size of the version string is returned.

/// @brief Get Kernel Program Binary
///
/// @details
///     - A valid kernel handle must be created with zeKernelCreate.
///     - Returns Intel Graphics Assembly (GEN ISA) format binary program data for kernel handle.
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
/// @returns
///     - ::ZE_RESULT_SUCCESS

///////////////////////////////////////////////////////////////////////////////
#ifndef ZEX_MEMORY_FREE_CALLBACK_EXT_NAME
/// @brief Memory Free Callback Extension Name
#define ZEX_MEMORY_FREE_CALLBACK_EXT_NAME "ZEX_extension_memory_free_callback"

///////////////////////////////////////////////////////////////////////////////
/// @brief Memory Free Callback Extension Version(s)
typedef enum _zex_memory_free_callback_ext_version_t {
    ZEX_MEMORY_FREE_CALLBACK_EXT_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZEX_MEMORY_FREE_CALLBACK_EXT_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZEX_MEMORY_FREE_CALLBACK_EXT_VERSION_FORCE_UINT32 = 0x7fffffff        ///< Value marking end of ZEX_MEMORY_FREE_CALLBACK_EXT_VERSION_* ENUMs

} zex_memory_free_callback_ext_version_t;

#ifndef ZEX_STRUCTURE_TYPE_MEMORY_FREE_CALLBACK_EXT_DESC
/// @brief stype for _zex_memory_free_callback_ext_desc_t
#endif

/**
 * @brief Callback function type for memory free events.
 *
 * This function is called when a memory free operation occurs.
 *
 * @param pUserData Pointer to user-defined data passed to the callback.
 */
typedef void (*zex_mem_free_callback_fn_t)(void *pUserData);

/**
 * @brief Descriptor for a memory free callback extension.
 *
 * This structure is used to specify a callback function that will be invoked when memory is freed.
 *
 * Members:
 * - stype: Specifies the type of this structure.
 * - pNext: Optional pointer to an extension-specific structure; must be null or point to a structure containing stype and pNext.
 * - pfnCallback: Callback function to be called when memory is freed.
 * - pUserData: Optional user data to be passed to the callback function.
 */
typedef struct _zex_memory_free_callback_ext_desc_t {
    ze_structure_type_ext_t stype;          ///< [in] type of this structure
    const void *pNext;                      ///< [in][optional] must be null or a pointer to an extension-specific
                                            ///< structure (i.e. contains stype and pNext).
    zex_mem_free_callback_fn_t pfnCallback; // [in] callback function to be called on memory free
    void *pUserData;                        // [in][optional] user data passed to callback
} zex_memory_free_callback_ext_desc_t;

/**
 * @brief Registers a callback to be invoked when memory is freed.
 *
 * This function allows the user to register a callback that will be called
 * whenever the specified memory is freed within the given context.
 *
 * @param hContext
 *        [in] Handle to the context in which the memory was allocated.
 * @param hFreeCallbackDesc
 *        [in] Pointer to a descriptor specifying the callback function and its parameters.
 * @param ptr
 *        [in] Pointer to the memory for which the free callback is to be registered.
 *
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_ARGUMENT
///         + `nullptr == hFreeCallbackDesc`
///         + `nullptr == ptr`
 *
 * @note The callback will be invoked when the specified memory is freed.
 */
ZE_APIEXPORT ze_result_t ZE_APICALL zexMemFreeRegisterCallbackExt(ze_context_handle_t hContext, zex_memory_free_callback_ext_desc_t *hFreeCallbackDesc, void *ptr);
#endif // ZEX_MEMORY_FREE_CALLBACK_EXT_NAME

#ifndef ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_NAME
/// @brief Get Kernel Program Binary experimental name
#define ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_NAME "ZE_intel_experimental_kernel_get_program_binary"
#endif // ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Intel Kernel Get Binary Extension Version(s)
typedef enum _ze_intel_kernel_get_binary_exp_version_t {
    ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_KERNEL_GET_PROGRAM_BINARY_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_intel_kernel_get_binary_exp_version_t;

ZE_APIEXPORT ze_result_t ZE_APICALL
zeIntelKernelGetBinaryExp(
    ze_kernel_handle_t hKernel, ///< [in] Kernel handle
    size_t *pSize,              ///< [in, out] pointer to variable with size of GEN ISA binary
    char *pKernelBinary         ///< [in,out] pointer to storage area for GEN ISA binary function
);

#ifndef ZE_INTEL_DRM_FORMAT_MODIFIER_EXP_NAME
/// @brief DRM format modifier extension name
#define ZE_INTEL_DRM_FORMAT_MODIFIER_EXP_NAME "ZE_intel_experimental_drm_format_modifier"
#endif // ZE_INTEL_DRM_FORMAT_MODIFIER_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief DRM format modifier extension Version(s)
typedef enum _ze_intel_drm_format_modifier_exp_version_t {
    ZE_INTEL_DRM_FORMAT_MODIFIER_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_INTEL_DRM_FORMAT_MODIFIER_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_INTEL_DRM_FORMAT_MODIFIER_EXP_VERSION_FORCE_UINT32 = 0x7fffffff
} ze_intel_drm_format_modifier_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image DRM format modifier properties
///
/// @details
///     - This structure may be passed as pNext member of ::ze_image_desc_t,
///       when using a DRM format modifier.
///     - Properties struct for providing user with the selected drm format modifier for the image
///     - This is useful if the application wants to export the image to another API that requires the DRM format modifier
///     - The application can query the chosen DRM format modifier for the image.
///     - The application can use this information to choose a DRM format modifier for the image during creation
typedef struct _ze_intel_image_selected_format_modifier_exp_properties_t {
    ze_structure_type_t stype;  ///< [in] type of this structure
    const void *pNext;          ///< [in][optional] must be null or a pointer to an extension-specific
                                ///< structure (i.e. contains stype and pNext).
    uint64_t drmFormatModifier; ///< [out] DRM format modifier
} ze_intel_image_selected_format_modifier_exp_properties_t;
///////////////////////////////////////////////////////////////////////////////
/// @brief Image DRM format modifier create list
///
/// @details
///     - This structure may be passed as pNext member of ::ze_image_desc_t,
///       when providing a list of DRM format modifiers to choose from during image creation.
///     - This is a descriptor for creating image with the specified list of drm format modifier
///     - If the user passes a list struct, then implementation chooses one from the list of drm modifiers as it sees fit.
///     - If user wants to pass a single drm modifier then they can set the drmFormatModifierCount to 1 and pass the single drm modifier in pDrmFormatModifiers
typedef struct _ze_intel_image_format_modifier_create_list_exp_desc_t {
    ze_structure_type_t stype;       ///< [in] type of this structure
    const void *pNext;               ///< [in][optional] must be null or a pointer to an extension-specific
                                     ///< structure (i.e. contains stype and pNext).
    uint32_t drmFormatModifierCount; ///< [in] number of DRM format modifiers in the list
    uint64_t *pDrmFormatModifiers;   ///< [in][range(0, drmFormatModifierCount)] array of DRM format modifiers
} ze_intel_image_format_modifier_create_list_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Image DRM format modifier import descriptor
///
/// @details
///     - This structure may be passed as pNext member of ::ze_image_desc_t,
///       when importing an image with a specific DRM format modifier.
///     - The pNext chain is setup accordingly in ze_image_desc_t prior to calling zeImageCreate API
typedef struct _ze_intel_image_format_modifier_import_exp_desc_t {
    ze_structure_type_t stype;  ///< [in] type of this structure
    const void *pNext;          ///< [in][optional] must be null or a pointer to an extension-specific
                                ///< structure (i.e. contains stype and pNext).
    uint64_t drmFormatModifier; ///< [in] DRM format modifier to use for the image
} ze_intel_image_format_modifier_import_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer DRM format modifier create list
///
/// @details
///     - This structure may be passed as pNext member of ::ze_device_mem_alloc_desc_t,
///       when providing a list of DRM format modifiers to choose from during buffer creation.
///     - This is a descriptor for creating buffer with the specified list of drm format modifier
///     - If the user passes a list struct, then implementation chooses one from the list of drm modifiers as it sees fit.
///     - If user wants to pass a single drm modifier then they can set the drmFormatModifierCount to 1 and pass the single drm modifier in pDrmFormatModifiers
///     - The pNext chain is setup accordingly in ze_device_mem_alloc_desc_t prior to calling zeMemAllocDevice API
typedef struct _ze_intel_mem_format_modifier_create_list_exp_desc_t {
    ze_structure_type_t stype;       ///< [in] type of this structure
    const void *pNext;               ///< [in][optional] must be null or a pointer to an extension-specific
                                     ///< structure (i.e. contains stype and pNext).
    uint32_t drmFormatModifierCount; ///< [in] number of DRM format modifiers in the list
    uint64_t *pDrmFormatModifiers;   ///< [in][range(0, drmFormatModifierCount)] array of DRM format modifiers
} ze_intel_mem_format_modifier_create_list_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer DRM format modifier import descriptor
///
/// @details
///     - This structure may be passed as pNext member of ::ze_device_mem_alloc_desc_t,
///       when importing a buffer with a specific DRM format modifier.
///     - This descriptor must be used in conjunction with ze_external_memory_import_fd_t. If not, implementation will return an error.
///     - The pNext chain is setup accordingly in ze_device_mem_alloc_desc_t prior to calling zeMemAllocDevice API
typedef struct _ze_intel_mem_format_modifier_import_exp_desc_t {
    ze_structure_type_t stype;  ///< [in] type of this structure
    const void *pNext;          ///< [in][optional] must be null or a pointer to an extension-specific
                                ///< structure (i.e. contains stype and pNext).
    uint64_t drmFormatModifier; ///< [in] DRM format modifier to use for the buffer
} ze_intel_mem_format_modifier_import_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Buffer DRM format modifier properties
///
/// @details
///     - This structure may be passed as pNext member of ::ze_memory_allocation_properties_t,
///       when querying the DRM format modifier of a buffer.
///     - Properties struct for providing user with the selected drm format modifier for the buffer
///     - This is useful if the application wants to export the buffer to another API that requires the DRM format modifier
///     - The application can query the chosen DRM format modifier for the buffer via zeMemGetAllocProperties API
typedef struct _ze_intel_mem_selected_format_modifier_exp_properties_t {
    ze_structure_type_t stype;  ///< [in] type of this structure
    const void *pNext;          ///< [in][optional] must be null or a pointer to an extension-specific
                                ///< structure (i.e. contains stype and pNext).
    uint64_t drmFormatModifier; ///< [out] DRM format modifier
} ze_intel_mem_selected_format_modifier_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Query for supported DRM format modifiers for a given image descriptor
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - This function can be used to query supported DRM format modifiers for a specific image description.
///     - User can use this API in two ways:
///         1. Set pCount to the address of a uint32_t with value 0 and pDrmFormatModifiers to nullptr
///            to query just the number of supported DRM format modifiers.
///         2. Set pCount to the address of a uint32_t with the number of elements in the pDrmFormatModifiers
///            array to retrieve the list of supported DRM format modifiers.
///     - The application can use the returned DRM format modifiers to:
///         1. Create L0 images with supported DRM format modifiers.
///         2. Compare with DRM format modifiers from other APIs (like Vulkan) to find common
///            modifiers that work for interop scenarios.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hDevice`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pCount`
///     - ::ZE_RESULT_ERROR_INVALID_IMAGE_DESC
///         + The image description doesn't match the device capabilities
ze_result_t ZE_APICALL
zeIntelImageGetFormatModifiersSupportedExp(
    ze_device_handle_t hDevice,        ///< [in] handle of the device
    const ze_image_desc_t *pImageDesc, ///< [in] pointer to image descriptor
    uint32_t *pCount,                  ///< [in,out] pointer to the number of DRM format modifiers.
                                       ///< if count is zero, then the driver shall update the value with the
                                       ///< total number of supported DRM format modifiers for the image format.
                                       ///< if count is greater than the number of supported DRM format modifiers,
                                       ///< then the driver shall update the value with the correct number of supported DRM format modifiers.
    uint64_t *pDrmFormatModifiers      ///< [in,out][optional][range(0, *pCount)] array of supported DRM format modifiers
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Query for supported DRM format modifiers for a memory allocation descriptor
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - This function can be used to query supported DRM format modifiers for a specific memory allocation description.
///     - User can use this API in two ways:
///         1. Set pCount to the address of a uint32_t with value 0 and pDrmFormatModifiers to nullptr
///            to query just the number of supported DRM format modifiers.
///         2. Set pCount to the address of a uint32_t with the number of elements in the pDrmFormatModifiers
///            array to retrieve the list of supported DRM format modifiers.
///     - The application can use the returned DRM format modifiers to:
///         1. Create L0 memory allocations with supported DRM format modifiers.
///         2. Compare with DRM format modifiers from other APIs (like Vulkan) to find common
///            modifiers that work for interop scenarios.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hContext`
ze_result_t ZE_APICALL
zeIntelMemGetFormatModifiersSupportedExp(
    ze_context_handle_t hContext,                  ///< [in] handle of the context
    const ze_device_mem_alloc_desc_t *pDeviceDesc, ///< [in] pointer to device memory allocation descriptor
    size_t size,                                   ///< [in] size in bytes to allocate
    size_t alignment,                              ///< [in] minimum alignment in bytes for the allocation
    ze_device_handle_t hDevice,                    ///< [in] handle of the device
    uint32_t *pCount,                              ///< [in,out] pointer to the number of DRM format modifiers.
                                                   ///< if count is zero, then the driver shall update the value with the
                                                   ///< total number of supported DRM format modifiers for the memory allocation.
                                                   ///< if count is greater than the number of supported DRM format modifiers,
                                                   ///< then the driver shall update the value with the correct number of supported DRM format modifiers.
    uint64_t *pDrmFormatModifiers                  ///< [in,out][optional][range(0, *pCount)] array of supported DRM format modifiers
);

/// @brief Get priority levels
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///    - Returns priority levels supported by the device
///    - lowestPriority reports the numerical value that corresponds to lowest queue priority
///    - highesPriority reports the numerical value that corresponds to highest queue priority
///    - Lower numbers indicate greater priorities
///    - The range of meaningful queue properties is represented by [*highestPriority, *lowestPriority]
///    - Priority passed upon queue creation would automatically clamp down or up to the nearest supported value
///    - 0 means default priority
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
ze_result_t ZE_APICALL zeDeviceGetPriorityLevels(
    ze_device_handle_t hDevice,
    int32_t *lowestPriority,
    int32_t *highestPriority);

///////////////////////////////////////////////////////////////////////////////
/// @brief Descriptor used for setting priority on command queues and immediate command lists.
/// This structure may be passed as pNext member of ::ze_command_queue_desc_t.
typedef struct _ze_queue_priority_desc_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    const void *pNext;             ///< [in][optional] must be null or a pointer to an extension-specific structure
    int priority;                  ///< [in] priority of the queue
} ze_queue_priority_desc_t;

/// @brief Get default context associated with default driver
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - Default context contains all devices within default driver instance
/// @returns
///     - Context handle associated with default driver
ZE_APIEXPORT ze_context_handle_t ZE_APICALL zerGetDefaultContext();

/// @brief Get Device Identifier
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - Returned identifier is a 32-bit unsigned integer that is unique to the driver.
///     - The identifier can be used then in zerTranslateIdentifierToDeviceHandle to get the device handle.
/// @returns
///     - 32-bit unsigned integer identifier
ZE_APIEXPORT uint32_t ZE_APICALL zerTranslateDeviceHandleToIdentifier(ze_device_handle_t hDevice); ///< [in] handle of the device

/// @brief Translate Device Identifier to Device Handle from default Driver
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///    - Returned device is associated to default driver handle.
/// @returns
///     - device handle associated with the identifier
ZE_APIEXPORT ze_device_handle_t ZE_APICALL zerTranslateIdentifierToDeviceHandle(uint32_t identifier); ///< [in] integer identifier of the device

///////////////////////////////////////////////////////////////////////////////
/// @brief Retrieves a string describing the last error code returned by the
///        default driver in the current thread.
///
/// @details
///     - String returned is thread local.
///     - String is only updated on calls returning an error, i.e., not on calls
///       returning ::ZE_RESULT_SUCCESS.
///     - String may be empty if driver considers error code is already explicit
///       enough to describe cause.
///     - Memory pointed to by ppString is owned by the driver.
///     - String returned is null-terminated.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_UNINITIALIZED
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == ppString`
ZE_APIEXPORT ze_result_t ZE_APICALL
zerGetLastErrorDescription(
    const char **ppString ///< [in,out] pointer to a null-terminated array of characters describing
                          ///< cause of error.
);

#if ZE_API_VERSION_CURRENT_M <= ZE_MAKE_VERSION(1, 13)

/// @brief Get default context associated with driver
///
/// @details
///     - The application may call this function from simultaneous threads.
///     - The implementation of this function should be lock-free.
///     - Default context contains all devices within driver instance
/// @returns
///     - Context handle associated with driver
ze_context_handle_t ZE_APICALL zeDriverGetDefaultContext(ze_driver_handle_t hDriver); ///> [in] handle of the driver

/// @brief Global device synchronization
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///    - Ensures that everything that was submitted to the device is completed.
///    - Ensures that all submissions in all queues on device are completed.
///    - It is not allowed to call this function while some command list are in graph capture mode.
///    - Returns error if error is detected during execution on device.
///    - Hangs indefinitely if GPU execution is blocked on non signaled event.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
ze_result_t ZE_APICALL zeDeviceSynchronize(ze_device_handle_t hDevice); ///> [in] handle of the device

/// @brief Append with arguments
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///    - Appends kernel to command list with arguments.
///    - Kernel object state is updated with new arguments, as if separate zeKernelSetArgumentValue were called.
///    - If argument is SLM (size), then SLM size in bytes for this resource is provided under pointer on specific index and its type is size_t.
///    - If argument is an immediate type (i.e. structure, non pointer type), then values under pointer must contain full size of immediate type.
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pArguments`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
typedef struct _ze_group_size_t {
    uint32_t groupSizeX; ///< [in] local work-group size in X dimension
    uint32_t groupSizeY; ///< [in] local work-group size in Y dimension
    uint32_t groupSizeZ; ///< [in] local work-group size in Z dimension

} ze_group_size_t;

ze_result_t ZE_APICALL zeCommandListAppendLaunchKernelWithArguments(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    ze_kernel_handle_t hKernel,            ///< [in] handle of the kernel object
    const ze_group_count_t groupCounts,    ///< [in] thread group counts
    const ze_group_size_t groupSizes,      ///< [in] thread group sizes
    void **pArguments,                     ///< [in] kernel arguments; pointer to list where each argument represents a pointer to the argument value on specific index
    const void *pNext,                     ///< [in][optional] extensions
    ze_event_handle_t hSignalEvent,        ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                ///< [in][optional] number of events to wait on before launching
    ze_event_handle_t *phWaitEvents);      ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait on before launching

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension descriptor for cooperative kernel launch via pNext chain.
/// @details
///     - This structure can be passed through pNext to zeCommandListAppendLaunchKernelWithParameters
typedef struct _ze_command_list_append_launch_kernel_param_cooperative_desc_t {
    ze_structure_type_ext_t stype; ///< [in] Type of this structure (e.g. ZE_STRUCTURE_TYPE_COMMAND_LIST_APPEND_PARAM_COOPERATIVE_DESC)
    const void *pNext;             ///< [in][optional] Pointer to the next extension-specific structure
    ze_bool_t isCooperative;       ///< [in] Indicates if the kernel should be launched as cooperative
} ze_command_list_append_launch_kernel_param_cooperative_desc_t;
/// @brief Append with parameters
///
/// @details
///    - The application may call this function from simultaneous threads.
///    - The implementation of this function should be lock-free.
///    - Appends kernel to command list with additional parameters via pNext chain.
///    - Allows passing core and extension descriptors (e.g. cooperative kernel).
///
/// @returns
///     - ::ZE_RESULT_SUCCESS
///     - ::ZE_RESULT_ERROR_DEVICE_LOST
///     - ::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY
///     - ::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
///     - ::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
///     - ::ZE_RESULT_ERROR_INVALID_NULL_HANDLE
///         + `nullptr == hCommandList`
///         + `nullptr == hKernel`
///     - ::ZE_RESULT_ERROR_INVALID_NULL_POINTER
///         + `nullptr == pGroupCounts`
///     - ::ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT
///     - ::ZE_RESULT_ERROR_INVALID_SIZE
///         + `(nullptr == phWaitEvents) && (0 < numWaitEvents)`
ZE_APIEXPORT ze_result_t ZE_APICALL zeCommandListAppendLaunchKernelWithParameters(
    ze_command_list_handle_t hCommandList, ///< [in] handle of the command list
    ze_kernel_handle_t hKernel,            ///< [in] handle of the kernel object
    const ze_group_count_t *pGroupCounts,  ///< [in] thread group launch arguments
    const void *pNext,                     ///< [in][optional] additional parameters (pNext chain)
    ze_event_handle_t hSignalEvent,        ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                ///< [in][optional] number of events to wait on before launching
    ze_event_handle_t *phWaitEvents        ///< [in][optional][range(0, numWaitEvents)] handle of the events to wait on before launching
);

#endif // ZE_API_VERSION_CURRENT_M <= ZE_MAKE_VERSION(1, 13)

#if defined(__cplusplus)
} // extern "C"
#endif

static const ze_device_mem_alloc_desc_t defaultIntelDeviceMemDesc = {
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, // stype
    nullptr,                                 // pNext
    ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED,    // flags
    0                                        // ordinal
};

static const ze_host_mem_alloc_desc_t defaultIntelHostMemDesc = {
    ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,                                             // stype
    nullptr,                                                                           // pNext
    ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED | ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT // flags
};

static const ze_command_queue_desc_t defaultIntelCommandQueueDesc = {
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,                                     // stype
    nullptr,                                                                  // pNext
    0,                                                                        // ordinal
    0,                                                                        // index
    ZE_COMMAND_QUEUE_FLAG_IN_ORDER | ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT, // flags
    ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,                                       // mode
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL                                          // priority
};

#if ZE_API_VERSION_CURRENT_M <= ZE_MAKE_VERSION(1, 13)

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME
/// @brief External Memory Mapping Extension Name
#define ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME "ZE_extension_external_memmap_sysmem"

///////////////////////////////////////////////////////////////////////////////
/// @brief External Memory Mapping Extension Version(s)
typedef enum _ze_external_memmap_sysmem_ext_version_t {
    ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_FORCE_UINT32 = 0x7fffffff        ///< Value marking end of ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_* ENUMs

} ze_external_memmap_sysmem_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Maps external system memory for an allocation
///
/// @details
///     - This structure may be passed to ::zeMemAllocHost, via the `pNext`
///       member of ::ze_host_mem_alloc_desc_t to map system memory for a host
///       allocation.
///     - The system memory pointer and size being mapped must be page aligned
///       based on the supported page sizes on the device.
typedef struct _ze_external_memmap_sysmem_ext_desc_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    const void *pNext;             ///< [in][optional] must be null or a pointer to an extension-specific
                                   ///< structure (i.e. contains stype and pNext).
    const void *pSystemMemory;     ///< [in] system memory pointer to map; must be page-aligned.
    const uint64_t size;           ///< [in] size of the system memory to map; must be page-aligned.

} ze_external_memmap_sysmem_ext_desc_t;
#endif // ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME

#endif // ZE_API_VERSION_CURRENT_M <= ZE_MAKE_VERSION(1, 13)
#endif
