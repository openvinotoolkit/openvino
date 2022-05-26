// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x) _x;
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) true

#define DEBUG_LOG(...)  \
    do {                                                                               \
        ::std::stringstream ss___;                                                     \
        ::ov::write_all_to_stream(ss___, "[ DEBUG ] ", __func__, ":", __LINE__, " ", __VA_ARGS__);                                 \
        std::cout << ss___.str() << std::endl;   \
    } while (0)

#else // !CPU_DEBUG_CAPS

#define CPU_DEBUG_CAP_ENABLE(_x)
#define CPU_DEBUG_CAPS_ALWAYS_TRUE(x) x

#define DEBUG_LOG(...)

#endif // CPU_DEBUG_CAPS
