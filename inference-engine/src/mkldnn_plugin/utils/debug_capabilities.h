// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS
#   define ENABLE_CPU_DEBUG_CAP(_x) _x;
#else
#   define ENABLE_CPU_DEBUG_CAP(_x)
#endif
