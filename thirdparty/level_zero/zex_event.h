// intel/compute-runtime 4df478c5139703c82e548a65eafbcc69923953ac
/*
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef _ZEX_EVENT_H
#define _ZEX_EVENT_H
#if defined(__cplusplus)
#pragma once
#endif

#include <level_zero/ze_api.h>

#include "zex_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZE_APIEXPORT ze_result_t ZE_APICALL
zexEventGetDeviceAddress(
    ze_event_handle_t event,
    uint64_t *completionValue,
    uint64_t *address);

// deprecated
ZE_APIEXPORT ze_result_t ZE_APICALL
zexCounterBasedEventCreate(
    ze_context_handle_t hContext,
    ze_device_handle_t hDevice,
    uint64_t *deviceAddress,
    uint64_t *hostAddress,
    uint64_t completionValue,
    const ze_event_desc_t *desc,
    ze_event_handle_t *phEvent);

ZE_APIEXPORT ze_result_t ZE_APICALL zexIntelAllocateNetworkInterrupt(ze_context_handle_t hContext, uint32_t &networkInterruptId);

ZE_APIEXPORT ze_result_t ZE_APICALL zexIntelReleaseNetworkInterrupt(ze_context_handle_t hContext, uint32_t networkInterruptId);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCounterBasedEventCreate2(ze_context_handle_t hContext, ze_device_handle_t hDevice, const zex_counter_based_event_desc_t *desc, ze_event_handle_t *phEvent);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCounterBasedEventGetIpcHandle(ze_event_handle_t hEvent, zex_ipc_counter_based_event_handle_t *phIpc);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCounterBasedEventOpenIpcHandle(ze_context_handle_t hContext, zex_ipc_counter_based_event_handle_t hIpc, ze_event_handle_t *phEvent);

ZE_APIEXPORT ze_result_t ZE_APICALL zexCounterBasedEventCloseIpcHandle(ze_event_handle_t hEvent);

ZE_APIEXPORT ze_result_t ZE_APICALL zexDeviceGetAggregatedCopyOffloadIncrementValue(ze_device_handle_t hDevice, uint32_t *incrementValue);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZEX_EVENT_H
