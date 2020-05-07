// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __NC_ENV_H_INCLUDED__
#define __NC_ENV_H_INCLUDED__

#include <stddef.h>
#include "mvnc.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// Returns 1 on success, 0 on failure.
MVNC_EXPORT_API int utf8_getenv_s(size_t bufferSize, char* buffer, const char* varName);

#ifdef __cplusplus
}
#endif

#endif  // __NC_ENV_H_INCLUDED__
