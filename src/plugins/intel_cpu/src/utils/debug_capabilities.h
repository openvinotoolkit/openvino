// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x) _x;
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

#else // !CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#endif // CPU_DEBUG_CAPS
