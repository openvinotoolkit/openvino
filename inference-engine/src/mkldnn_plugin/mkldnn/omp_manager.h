// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file omp_manager.h
 */
#pragma once

#ifdef _WIN32
    #include "mkldnn/os/win/win_omp_manager.h"
#elif defined(__APPLE__)
    #include "mkldnn/os/osx/osx_omp_manager.h"
#else
    #include "mkldnn/os/lin/lin_omp_manager.h"
#endif
