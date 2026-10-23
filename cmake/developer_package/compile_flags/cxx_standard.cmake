# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Fails fast when the active compiler is too old to parse C++20 (the project default),
# or whatever higher standard CMAKE_CXX_STANDARD is overridden to from the command line.
# Does nothing when CMAKE_CXX_STANDARD is explicitly lowered below 20.

if(NOT CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    return()
endif()
# OpenVINO has no C++20 named modules; this scanning only costs a version-matched
# clang-scan-deps (CMP0155/CMake 3.28+) that may not exist in every toolchain/image.
if(NOT DEFINED CMAKE_CXX_SCAN_FOR_MODULES)
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
endif()

# CVS-190125: raise the GNU floor (currently 10) once OpenVINO drops the OSes that
# ship it (e.g. RHEL 8).
set(_ov_cxx20_floor_GNU 10)
set(_ov_cxx20_floor_Clang 14)
set(_ov_cxx20_floor_AppleClang 13)
set(_ov_cxx20_floor_MSVC 19.28)
set(_ov_cxx20_floor_IntelLLVM 2024.2)

if(DEFINED _ov_cxx20_floor_${CMAKE_CXX_COMPILER_ID})
    set(_ov_cxx20_floor "${_ov_cxx20_floor_${CMAKE_CXX_COMPILER_ID}}")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS _ov_cxx20_floor)
        message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not fully support C++20. "
            "Minimum: ${CMAKE_CXX_COMPILER_ID} ${_ov_cxx20_floor}")
    endif()

    unset(_ov_cxx20_floor)
else()
    message(WARNING
        "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} was requested, but ${CMAKE_CXX_COMPILER_ID} "
        "has no verified minimum version in cxx_standard.cmake. Proceeding unchecked.")
endif()

unset(_ov_cxx20_floor_GNU)
unset(_ov_cxx20_floor_Clang)
unset(_ov_cxx20_floor_AppleClang)
unset(_ov_cxx20_floor_MSVC)
unset(_ov_cxx20_floor_IntelLLVM)
