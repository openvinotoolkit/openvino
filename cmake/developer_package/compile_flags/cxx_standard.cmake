# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Verifies the toolchain can build C++20 when CMAKE_CXX_STANDARD requests it.

if(NOT CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    return()
endif()

if(NOT DEFINED OV_CXX_STANDARD_FALLBACK)
    set(OV_CXX_STANDARD_FALLBACK 17)
endif()
# OpenVINO has no C++20 named modules; this scanning only costs a version-matched
# clang-scan-deps (CMP0155/CMake 3.28+) that may not exist in every toolchain/image.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28 AND NOT DEFINED CMAKE_CXX_SCAN_FOR_MODULES)
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
endif()

# CVS-190125: raise the GNU floor (currently 10) once OpenVINO drops the OSes that
# ship it (e.g. RHEL 8) as part of the 2027.0 C++20 default switch.
set(_ov_cxx20_floor_GNU 10)
set(_ov_cxx20_floor_Clang 14)
set(_ov_cxx20_floor_AppleClang 13)
set(_ov_cxx20_floor_MSVC 19.28)
set(_ov_cxx20_floor_IntelLLVM 2024.2)

set(_ov_cxx20_error "")

if(DEFINED _ov_cxx20_floor_${CMAKE_CXX_COMPILER_ID})
    set(_ov_cxx20_floor "${_ov_cxx20_floor_${CMAKE_CXX_COMPILER_ID}}")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS _ov_cxx20_floor)
        string(CONCAT _ov_cxx20_error
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not fully support C++20. "
            "Minimum: ${CMAKE_CXX_COMPILER_ID} ${_ov_cxx20_floor}")
    endif()

    unset(_ov_cxx20_floor)
else()
    message(WARNING
        "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} was requested, but ${CMAKE_CXX_COMPILER_ID} "
        "has no verified minimum version in cxx_standard.cmake. Proceeding unchecked.")
endif()

# The compiler version does not guarantee the C++20 library (e.g. clang with an old libstdc++).
if(NOT _ov_cxx20_error)
    include(CheckCXXSourceCompiles)

    check_cxx_source_compiles("
        #include <bit>
        #include <concepts>
        #include <span>
        template <std::integral T>
        int f(std::span<const T> s) { return std::countl_zero(static_cast<unsigned>(s.size())); }
        int main() {
            const int a[1] = {0};
            return f<int>(a) > 100;
        }"
        OV_CXX20_LIBRARY_AVAILABLE)

    if(NOT OV_CXX20_LIBRARY_AVAILABLE)
        string(CONCAT _ov_cxx20_error
            "The C++ standard library of ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "lacks <bit>/<concepts>/<span> required for C++20")
    endif()
endif()

if(_ov_cxx20_error)
    # only a defaulted standard may fall back, an explicit request must fail
    if(OV_CXX_STANDARD_IS_DEFAULT)
        message(STATUS "${_ov_cxx20_error}. Falling back to C++${OV_CXX_STANDARD_FALLBACK}")
        set(CMAKE_CXX_STANDARD ${OV_CXX_STANDARD_FALLBACK})
    else()
        message(FATAL_ERROR
            "${_ov_cxx20_error} (CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} was requested)")
    endif()
endif()

unset(_ov_cxx20_error)
unset(_ov_cxx20_floor_GNU)
unset(_ov_cxx20_floor_Clang)
unset(_ov_cxx20_floor_AppleClang)
unset(_ov_cxx20_floor_MSVC)
unset(_ov_cxx20_floor_IntelLLVM)
