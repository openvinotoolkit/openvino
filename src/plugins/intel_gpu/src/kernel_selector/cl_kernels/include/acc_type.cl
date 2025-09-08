// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Defines default accumulator type.
// TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
// TODO: Get rid of this include and generate proper accumulator type on host (when needed)
#if !defined(ACCUMULATOR_TYPE)
    #define ACCUMULATOR_TYPE float
    #define TO_ACCUMULATOR_TYPE(v) (float)(v)
    #define ACCUMULATOR_TYPE_ZERO 0.0f
#endif
