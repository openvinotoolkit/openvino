// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MVNC_XLINK_DEVICE_H
#define MVNC_XLINK_DEVICE_H

#include "mvnc.h"
#include "watchdog.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define WATCHDOG_MAX_PING_INTERVAL_MS 1000

wd_error_t xlink_device_create(WdDeviceHndl_t** out_deviceHandle, devicePrivate_t* pDevice);
void xlink_device_destroy(WdDeviceHndl_t* deviceHandle);

#ifdef __cplusplus
}
#endif

#endif
