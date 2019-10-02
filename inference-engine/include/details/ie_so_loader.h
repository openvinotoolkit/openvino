// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared objects
 * @file ie_so_loader.h
 */
#pragma once

#ifndef _WIN32
    #include "os/lin_shared_object_loader.h"
#else
    #include "os/win_shared_object_loader.h"
#endif
