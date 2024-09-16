// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
#    if (__cplusplus <= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG <= 201703L) && (_MSC_VER <= 1913))
#        if __has_include(<experimental/filesystem>)
#        include <experimental/filesystem>
#        define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#        define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
namespace std_fs = std::experimental::filesystem;
#        endif
#    elif __has_include(<filesystem>)
#            include <filesystem>
namespace std_fs = std::filesystem;
#    else
#       error "No std filesystem lib avaliable!"
#    endif
#endif
