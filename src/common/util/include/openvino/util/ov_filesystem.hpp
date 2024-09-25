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

#if defined(_MSC_VER) && defined(CPP_VER_17)
#    define HAS_FILESYSTEM 1
#elif defined(_MSC_VER) && defined(CPP_VER_11)
#    define HAS_EXP_FILESYSTEM 1
#    define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#    define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
#elif defined(__has_include)
#    if defined(CPP_VER_17) && (__has_include(<filesystem>)) && (!__has_include(<experimental/filesystem>))
#        define HAS_FILESYSTEM 1
#    elif defined(CPP_VER_11) && (__has_include(<experimental/filesystem>))
#        define HAS_EXP_FILESYSTEM 1
#    endif
#endif

#if !defined(HAS_FILESYSTEM) && !defined(HAS_EXP_FILESYSTEM)
#    error "Neither #include <filesystem> nor #include <experimental/filesystem> is available."
#elif defined(HAS_FILESYSTEM)
    #include <filesystem>
    namespace std_fs = std::filesystem;
#elif defined(HAS_EXP_FILESYSTEM)
    #include <experimental/filesystem>
    namespace std_fs = std::experimental::filesystem;
#endif

// #if !defined(__EMSCRIPTEN__) && !defined(__ANDROID__)
// #    if (__cplusplus <= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG <= 201703L) && (_MSC_VER <= 1913))
// #        if __has_include(<experimental/filesystem>)
// #        include <experimental/filesystem>
// #        define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
// #        define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
// namespace std_fs = std::experimental::filesystem;
// #        endif
// #    elif __has_include(<filesystem>)
// #            include <filesystem>
// namespace std_fs = std::filesystem;
// #    else
// #       error "No std filesystem lib avaliable!"
// #    endif
// #endif

// // We haven't checked which filesystem to include yet
// #ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

// // Check for feature test macro for <filesystem>
// #   if defined(__cpp_lib_filesystem)
// #       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0

// // Check for feature test macro for <experimental/filesystem>
// #   elif defined(__cpp_lib_experimental_filesystem)
// #       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// // We can't check if headers exist...
// // Let's assume experimental to be safe
// #   elif !defined(__has_include)
// #       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// // Check if the header "<filesystem>" exists
// #   elif __has_include(<filesystem>) && ((__cplusplus >= 201703L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L) && (_MSC_VER >= 1913)))

// // If we're compiling on Visual Studio and are not compiling with C++17, we need to use experimental
// #       ifdef _MSC_VER

// // Check and include header that defines "_HAS_CXX17"
// #           if __has_include(<yvals_core.h>)
// #               include <yvals_core.h>

// // Check for enabled C++17 support
// #               if defined(_HAS_CXX17) && _HAS_CXX17
// // We're using C++17, so let's use the normal version
// #                   define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
// #               endif
// #           endif

// // If the marco isn't defined yet, that means any of the other VS specific checks failed, so we need to use
// experimental #           ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// #               define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
// #           endif

// // Not on Visual Studio. Let's use the normal version
// #       else // #ifdef _MSC_VER
// #           define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
// #       endif

// // Check if the header "<filesystem>" exists
// #   elif __has_include(<experimental/filesystem>)
// #       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// // Fail if neither header is available with a nice error message
// #   else
// #       error Could not find system header "<filesystem>" or "<experimental/filesystem>"
// #   endif

// // We priously determined that we need the exprimental version
// #   if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// // Include it
// #       define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
// #       define _LIBCPP_NO_EXPERIMENTAL_DEPRECATION_WARNING_FILESYSTEM
// #       include <experimental/filesystem>

// // We need the alias from std::experimental::filesystem to std::filesystem
// namespace std_fs = std::experimental::filesystem;

// // We have a decent compiler and can use the normal version
// #   else
// // Include it
// #       include <filesystem>
// namespace std_fs = std::filesystem;
// #   endif

// #endif // #ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
