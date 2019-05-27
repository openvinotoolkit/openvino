// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MVNC_WATCHDOG_H
#define MVNC_WATCHDOG_H

#include <mvnc.h>
#ifdef __cplusplus
# define WD_API  extern "C"
# else
# define WD_API
#endif

/**
* @brief default ping interval is 1 second
*/
#define WATCHDOG_PING_INTERVAL_MS 1000

typedef struct wd_context_tag {
    void * opaque;
} wd_context;

typedef enum {
    WD_ERRNO = 0,
    WD_NOTINITIALIZED,
    WD_DUPLICATE,
    WD_FAIL
} wd_error_t;

/**
 * @brief initializes watchdog context, required to be called before any other WD API calls
 * @return
 */
WD_API wd_error_t watchdog_init_context(wd_context *ctx);

/**
 * @brief creates watchdog thread, if not created, and registers new watchee device, and initialise opaque handle to it
 * @param d - newly connected device descriptor
 * @return
 */
WD_API wd_error_t watchdog_register_device(wd_context *ctx, devicePrivate_t *d);

/**
 * @brief remove watch_dog device from the list, and might stop watchdog worker thread
 * @return result of operation
 */
WD_API wd_error_t watchdog_unregister_device(wd_context *ctx);


#endif  // MVNC_WATCHDOG_H
