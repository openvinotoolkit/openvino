// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MVNC_WATCHDOG_H
#define MVNC_WATCHDOG_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct _WatchdogHndl_t WatchdogHndl_t;

typedef struct _WdDeviceHndl_t {
    void* m_device;
} WdDeviceHndl_t;

typedef enum {
    WD_ERRNO = 0,
    WD_NOTINITIALIZED,
    WD_FAIL
} wd_error_t;

wd_error_t watchdog_create(WatchdogHndl_t** out_watchdogHndl);
void watchdog_destroy(WatchdogHndl_t* watchdogHndl);

/**
 * @brief Creates watchdog thread, if not created, and registers new watchee device, and initialise opaque handle to it.
 *        To avoid a memory leak, the registered device must be unregister with watchdog_unregister_device().
 * @param deviceHandle - newly connected device descriptor
 * @return
 */
wd_error_t watchdog_register_device(WatchdogHndl_t* watchdogHndl, WdDeviceHndl_t* deviceHandle);

/**
 * @brief remove watch_dog device from the list, and might stop watchdog worker thread
 * @return result of operation
 */
wd_error_t watchdog_unregister_device(WatchdogHndl_t* watchdogHndl, WdDeviceHndl_t* deviceHandle);

#ifdef __cplusplus
}
#endif

#endif  // MVNC_WATCHDOG_H
