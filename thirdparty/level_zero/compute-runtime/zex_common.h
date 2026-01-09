// intel/compute-runtime 4df478c5139703c82e548a65eafbcc69923953ac
/*
 * Copyright (C) 2022-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_COMMON_H
#define _ZEX_COMMON_H
#if defined(__cplusplus)
#pragma once
#endif
#include "ze_stypes.h"
#include <ze_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of command list object
typedef ze_command_list_handle_t zex_command_list_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of event object
typedef ze_event_handle_t zex_event_handle_t;

#define ZEX_BIT(_i) (1 << _i)

typedef uint32_t zex_mem_action_scope_flags_t;
typedef enum _zex_mem_action_scope_flag_t {
    ZEX_MEM_ACTION_SCOPE_FLAG_SUBDEVICE = ZEX_BIT(0),
    ZEX_MEM_ACTION_SCOPE_FLAG_DEVICE = ZEX_BIT(1),
    ZEX_MEM_ACTION_SCOPE_FLAG_HOST = ZEX_BIT(2),
    ZEX_MEM_ACTION_SCOPE_FLAG_FORCE_UINT32 = 0x7fffffff
} zex_mem_action_scope_flag_t;

typedef uint32_t zex_wait_on_mem_action_flags_t;
typedef enum _zex_wait_on_mem_action_flag_t {
    ZEX_WAIT_ON_MEMORY_FLAG_EQUAL = ZEX_BIT(0),
    ZEX_WAIT_ON_MEMORY_FLAG_NOT_EQUAL = ZEX_BIT(1),
    ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN = ZEX_BIT(2),
    ZEX_WAIT_ON_MEMORY_FLAG_GREATER_THAN_EQUAL = ZEX_BIT(3),
    ZEX_WAIT_ON_MEMORY_FLAG_LESSER_THAN = ZEX_BIT(4),
    ZEX_WAIT_ON_MEMORY_FLAG_LESSER_THAN_EQUAL = ZEX_BIT(5),
    ZEX_WAIT_ON_MEMORY_FLAG_FORCE_UINT32 = 0x7fffffff
} zex_wait_on_mem_action_flag_t;

typedef struct _zex_wait_on_mem_desc_t {
    zex_wait_on_mem_action_flags_t actionFlag;
    zex_mem_action_scope_flags_t waitScope;
} zex_wait_on_mem_desc_t;

typedef struct _zex_write_to_mem_desc_t {
    zex_mem_action_scope_flags_t writeScope;
} zex_write_to_mem_desc_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_SYNCHRONIZED_DISPATCH_EXP_NAME
/// @brief Synchronized Dispatch extension name
#define ZE_SYNCHRONIZED_DISPATCH_EXP_NAME "ZE_experimental_synchronized_dispatch"
#endif // ZE_SYNCHRONIZED_DISPATCH_EXP_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Synchronized Dispatch extension version(s)
typedef enum _ze_synchronized_dispatch_exp_version_t {
    ZE_SYNCHRONIZED_DISPATCH_EXP_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZE_SYNCHRONIZED_DISPATCH_EXP_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZE_SYNCHRONIZED_DISPATCH_EXP_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_synchronized_dispatch_exp_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported synchronized dispatch flags
typedef uint32_t ze_synchronized_dispatch_exp_flags_t;
typedef enum _ze_synchronized_dispatch_exp_flag_t {
    ZE_SYNCHRONIZED_DISPATCH_DISABLED_EXP_FLAG = ZE_BIT(0), ///< Non-synchronized dispatch. Must synchronize only with other synchronized dispatches
    ZE_SYNCHRONIZED_DISPATCH_ENABLED_EXP_FLAG = ZE_BIT(1),  ///< Synchronized dispatch. Must synchronize with all synchronized and non-synchronized dispatches
    ZE_SYNCHRONIZED_DISPATCH_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_synchronized_dispatch_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_synchronized_dispatch_exp_desc_t
typedef struct _ze_synchronized_dispatch_exp_desc_t ze_synchronized_dispatch_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Command queue or command list descriptor for synchronized dispatch. This structure may be
///        passed as pNext member of ::ze_command_queue_desc_t. or ::ze_command_list_desc_t.
typedef struct _ze_synchronized_dispatch_exp_desc_t {
    ze_structure_type_ext_t stype;              ///< [in] type of this structure
    const void *pNext;                          ///< [in][optional] must be null or a pointer to an extension-specific
                                                ///< structure (i.e. contains stype and pNext).
    ze_synchronized_dispatch_exp_flags_t flags; ///< [in] mode flags.
                                                ///< must be valid value of ::ze_synchronized_dispatch_exp_flag_t

} ze_synchronized_dispatch_exp_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_intel_media_communication_desc_t
typedef struct _ze_intel_media_communication_desc_t ze_intel_media_communication_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief ze_intel_media_communication_desc_t
typedef struct _ze_intel_media_communication_desc_t {
    ze_structure_type_ext_t stype;          ///< [in] type of this structure
    void *pNext;                            ///< [in][optional] must be null or a pointer to an extension-specific, this will be used to extend this in future
    void *controlSharedMemoryBuffer;        ///< [in] control shared memory buffer pointer, must be USM address
    uint32_t controlSharedMemoryBufferSize; ///< [in] control shared memory buffer size
    void *controlBatchBuffer;               ///< [in] control batch buffer pointer, must be USM address
    uint32_t controlBatchBufferSize;        ///< [in] control batch buffer size
} ze_intel_media_communication_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_intel_media_doorbell_handle_desc_t
typedef struct _ze_intel_media_doorbell_handle_desc_t ze_intel_media_doorbell_handle_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief ze_intel_media_doorbell_handle_desc_t
/// @details Handle of the doorbell. This structure is passed as argument of zeIntelMediaCommunicationCreate and zeIntelMediaCommunicationDestroy
typedef struct _ze_intel_media_doorbell_handle_desc_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    void *pNext;                   ///< [in][optional] must be null or a pointer to an extension-specific, this will be used to extend this in future
    void *doorbell;                ///< [in,out] handle of the doorbell
} ze_intel_media_doorbell_handle_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported device media flags
typedef uint32_t ze_intel_device_media_exp_flags_t;
typedef enum _ze_intel_device_media_exp_flag_t {
    ZE_INTEL_DEVICE_MEDIA_SUPPORTS_ENCODING_EXP_FLAG = ZE_BIT(0), ///< Supports encoding
    ZE_INTEL_DEVICE_MEDIA_SUPPORTS_DECODING_EXP_FLAG = ZE_BIT(1), ///< Supports decoding
    ZE_INTEL_DEVICE_MEDIA_EXP_FLAG_FORCE_UINT32 = 0x7fffffff
} ze_intel_device_media_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Forward-declare ze_intel_device_media_exp_properties_t
typedef struct _ze_intel_device_media_exp_properties_t ze_intel_device_media_exp_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief  May be passed to ze_device_properties_t through pNext.
typedef struct _ze_intel_device_media_exp_properties_t {
    ze_structure_type_ext_t stype;           ///< [in] type of this structure
    const void *pNext;                       ///< [in][optional] must be null or a pointer to an extension-specific
    ze_intel_device_media_exp_flags_t flags; ///< [out] device media flags
    uint32_t numEncoderCores;                ///< [out] number of encoder cores
    uint32_t numDecoderCores;                ///< [out] number of decoder cores
} ze_intel_device_media_exp_properties_t;

#ifndef ZEX_COUNTER_BASED_EVENT_EXT_NAME
/// @brief Counter Based Event Extension Name
#define ZEX_COUNTER_BASED_EVENT_EXT_NAME "ZEX_counter_based_event"
#endif // ZEX_COUNTER_BASED_EVENT_EXT_NAME

///////////////////////////////////////////////////////////////////////////////
/// @brief Counter Based Event Extension Version(s)
typedef enum _zex_counter_based_event_version_t {
    ZEX_COUNTER_BASED_EVENT_VERSION_1_0 = ZE_MAKE_VERSION(1, 0),     ///< version 1.0
    ZEX_COUNTER_BASED_EVENT_VERSION_CURRENT = ZE_MAKE_VERSION(1, 0), ///< latest known version
    ZEX_COUNTER_BASED_EVENT_VERSION_FORCE_UINT32 = 0x7fffffff

} zex_counter_based_event_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief IPC handle to counter based event
typedef struct _zex_ipc_counter_based_event_handle_t {
    char data[ZE_MAX_IPC_HANDLE_SIZE]; ///< [out] Opaque data representing an IPC handle
} zex_ipc_counter_based_event_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported event flags for defining counter-based event
typedef uint32_t zex_counter_based_event_exp_flags_t;
typedef enum _zex_counter_based_event_exp_flag_t {
    ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE = ZE_BIT(0),               ///< Counter-based event is used for immediate command lists (default)
    ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE = ZE_BIT(1),           ///< Counter-based event is used for non-immediate command lists
    ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE = ZE_BIT(2),            ///< Signals and waits are also visible to host
    ZEX_COUNTER_BASED_EVENT_FLAG_IPC = ZE_BIT(3),                     ///< Event can be shared across processes for waiting
    ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP = ZE_BIT(4),        ///< Event contains kernel timestamps
    ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_MAPPED_TIMESTAMP = ZE_BIT(5), ///< Event contains kernel timestamps synchronized to host time domain.
                                                                      ///< Cannot be combined with::ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP
    ZEX_COUNTER_BASED_EVENT_FLAG_GRAPH_EXTERNAL_EVENT = ZE_BIT(6),    ///< Event when is used in graph record & replay, can be used outside
                                                                      ///< recorded graph for synchronization (using as wait event or for host synchronization)
    ZEX_COUNTER_BASED_EVENT_FLAG_FORCE_UINT32 = 0x7fffffff

} zex_counter_based_event_exp_flag_t;

typedef struct _zex_counter_based_event_desc_t {
    ze_structure_type_ext_t stype;             ///< [in] type of this structure
    const void *pNext;                         ///< [in][optional] must be null or a pointer to an extension-specific
    zex_counter_based_event_exp_flags_t flags; ///< [in] counter based event flags.
                                               ///< Must be 0 (default) or a valid combination of ::zex_counter_based_event_exp_flag_t
    ze_event_scope_flags_t signalScope;        ///< [in] defines the scope of relevant cache hierarchies to flush on a
                                               ///< signal action before the event is triggered.
                                               ///< must be 0 (default) or a valid combination of ::ze_event_scope_flag_t;
                                               ///< default behavior is synchronization within the command list only, no
                                               ///< additional cache hierarchies are flushed.
    ze_event_scope_flags_t waitScope;          ///< [in] defines the scope of relevant cache hierarchies to invalidate on
                                               ///< a wait action after the event is complete.
                                               ///< must be 0 (default) or a valid combination of ::ze_event_scope_flag_t;
                                               ///< default behavior is synchronization within the command list only, no
                                               ///< additional cache hierarchies are invalidated.
} zex_counter_based_event_desc_t;

static const zex_counter_based_event_desc_t defaultIntelCounterBasedEventDesc = {
    ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC, // stype
    nullptr,                                // pNext
    ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE |
        ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE |
        ZEX_COUNTER_BASED_EVENT_FLAG_HOST_VISIBLE, // flags
    ZE_EVENT_SCOPE_FLAG_HOST,                      // signalScope
    ZE_EVENT_SCOPE_FLAG_DEVICE                     // waitScope
};

///////////////////////////////////////////////////////////////////////////////
/// @brief Initial Counter Based Event synchronization parameters. This structure may be
///        passed as pNext member of ::zex_counter_based_event_desc_t.
typedef struct _zex_counter_based_event_external_sync_alloc_properties_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    const void *pNext;             ///< [in][optional] must be null or a pointer to an extension-specific
    uint64_t *deviceAddress;       ///< [in] device address for external synchronization allocation
    uint64_t *hostAddress;         ///< [in] host address for external synchronization allocation
    uint64_t completionValue;      ///< [in] completion value for external synchronization allocation
} zex_counter_based_event_external_sync_alloc_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Initial Counter Based Event synchronization parameters. This structure may be
///        passed as pNext member of ::zex_counter_based_event_desc_t.
typedef struct _zex_counter_based_event_external_storage_properties_t {
    ze_structure_type_ext_t stype; ///< [in] type of this structure
    const void *pNext;             ///< [in][optional] must be null or a pointer to an extension-specific
    uint64_t *deviceAddress;       ///< [in] device address that would be updated with atomic_add upon signaling of this event, must be device USM memory
    uint64_t incrementValue;       ///< [in] value which would by atomically added upon each completion
    uint64_t completionValue;      ///< [in] final completion value, when value under deviceAddress is equal or greater then this value then event is considered as completed
} zex_counter_based_event_external_storage_properties_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_COMMON_EXTENDED_H
