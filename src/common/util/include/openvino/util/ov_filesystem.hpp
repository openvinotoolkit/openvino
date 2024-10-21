// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !(defined(_MSC_VER) && __cplusplus == 199711L)
#    if __cplusplus >= 201103L
#        define CPP_VER_11
#        if __cplusplus >= 201402L
#            define CPP_VER_14
#            if __cplusplus >= 201703L
#                define CPP_VER_17
#                if __cplusplus >= 202002L
#                    define CPP_VER_20
#                endif
#            endif
#        endif
#    endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
#    if _MSVC_LANG >= 201103L
#        define CPP_VER_11
#        if _MSVC_LANG >= 201402L
#            define CPP_VER_14
#            if _MSVC_LANG >= 201703L
#                define CPP_VER_17
#                if _MSVC_LANG >= 202002L
#                    define CPP_VER_20
#                endif
#            endif
#        endif
#    endif
#endif

#if defined(_MSC_VER) && defined(CPP_VER_11)
#    define HAS_EXP_FILESYSTEM 1
#    define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#    define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#elif defined(ANDROID) || defined(__ANDROID__)
#    define HAS_EXP_FILESYSTEM 1
#    define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#    define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#elif defined(__has_include)
#    if defined(CPP_VER_17) && (__has_include(<filesystem>)) && (!__has_include(<experimental/filesystem>))
#        define HAS_FILESYSTEM 1
#    elif defined(CPP_VER_11) && (__has_include(<experimental/filesystem>))
#        define HAS_EXP_FILESYSTEM 1
#        define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#        define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#    endif
#endif

#if !defined(HAS_FILESYSTEM) && !defined(HAS_EXP_FILESYSTEM)
#    error "Neither #include <filesystem> nor #include <experimental/filesystem> is available."
#elif defined(HAS_FILESYSTEM)
#    include <filesystem>
namespace std_fs = std::filesystem;
#elif defined(HAS_EXP_FILESYSTEM)
#    include <experimental/filesystem>
namespace std_fs = std::experimental::filesystem;
#endif
