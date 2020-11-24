// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file to include mapped file implementation
 * 
 * @file ie_mmap.hpp
 */
#pragma once

#ifdef _WIN32
#include "ie_mmap_windows.hpp"
#else
#include "ie_mmap_linux.hpp"
#endif
