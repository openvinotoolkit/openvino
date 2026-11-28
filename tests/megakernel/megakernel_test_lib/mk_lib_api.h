// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "exportApi.h"
#include "iMegakernelRuntime.h"

extern "C" TEST_MEGAKERNEL_EXPORT mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime();
extern "C" TEST_MEGAKERNEL_EXPORT void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime);