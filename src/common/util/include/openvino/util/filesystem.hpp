// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "util.hpp"

#if defined(_MSC_VER) && defined(OPENVINO_CPP_VER_11)
#    define OPENVINO_HAS_EXP_FILESYSTEM
#    define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#    define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#elif defined(__has_include)
#    if defined(OPENVINO_CPP_VER_17) && (__has_include(<filesystem>)) && (!__has_include(<experimental/filesystem>))
#        define OPENVINO_HAS_FILESYSTEM
#    elif defined(OPENVINO_CPP_VER_11) && (__has_include(<experimental/filesystem>))
#        define OPENVINO_HAS_EXP_FILESYSTEM
#        define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#        define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#    endif
#endif

#if defined(OPENVINO_HAS_FILESYSTEM)
#    include <filesystem>
#elif defined(OPENVINO_HAS_EXP_FILESYSTEM)
#    include <experimental/filesystem>
#else
#    error "Neither #include <filesystem> nor #include <experimental/filesystem> is available."
#endif
