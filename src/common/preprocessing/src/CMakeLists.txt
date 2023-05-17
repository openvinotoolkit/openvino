# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "openvino_gapi_preproc")

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

#
# Provides compilation flags for AVX512F, AVX512BW, AVX512DQ
# instructions support depending on an OS and a compiler
#
macro(ie_avx512_core_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(${flags} /arch:AVX512)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            set(${flags} /QxCORE-AVX512)
        else()
            set(${flags} -xCORE-AVX512)
        endif()
    elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        set(${flags} -mavx512f -mavx512bw -mavx512dq -mfma)
        if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
            list(APPEND ${flags} -Wno-error=maybe-uninitialized -Wno-maybe-uninitialized)
        endif()
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endmacro()

if(ENABLE_SSE42)
    file(GLOB SSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp)
    file(GLOB SSE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp)

    list(APPEND LIBRARY_HEADERS ${SSE_HEADERS})
    list(APPEND LIBRARY_SRC ${SSE_SRC})

    ie_sse42_optimization_flags(sse4_2_flags)
    set_source_files_properties(${SSE_SRC} PROPERTIES COMPILE_OPTIONS "${sse4_2_flags}")
    add_definitions(-DHAVE_SSE=1)
endif()

if(ENABLE_AVX2)
    file(GLOB AVX2_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_avx2/*.cpp)
    file(GLOB AVX2_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_avx2/*.hpp)

    list(APPEND LIBRARY_HEADERS ${AVX2_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX2_SRC})

    ie_avx2_optimization_flags(avx2_flags)
    set_source_files_properties(${AVX2_SRC} PROPERTIES COMPILE_OPTIONS "${avx2_flags}")
    add_definitions(-DHAVE_AVX2=1)
endif()


# Workaround for GCC version 5.4 and 5.5 bugs in Debug configuration.
if (CMAKE_COMPILER_IS_GNUCXX AND
    (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 5.5) AND
    (CMAKE_BUILD_TYPE STREQUAL Debug))
    set(GNU_5_DEBUG_CASE ON)
endif()

if(ENABLE_AVX512F AND NOT GNU_5_DEBUG_CASE)
    file(GLOB AVX512_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_avx512/*.cpp)
    file(GLOB AVX512_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_avx512/*.hpp)

    list(APPEND LIBRARY_HEADERS ${AVX512_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX512_SRC})

    ie_avx512_core_optimization_flags(avx512_flags)
    set_source_files_properties(${AVX512_SRC} PROPERTIES COMPILE_OPTIONS "${avx512_flags}")
    add_definitions(-DHAVE_AVX512=1)
endif()

if(ARM OR AARCH64)
    ie_arm_neon_optimization_flags(neon_flags)

    file(GLOB NEON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/arm_neon/*.hpp)
    file(GLOB NEON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/arm_neon/*.cpp)

    list(APPEND LIBRARY_HEADERS ${NEON_HEADERS})
    list(APPEND LIBRARY_SRC ${NEON_SRC})

    if(neon_flags)
        set_source_files_properties(${NEON_SRC} PROPERTIES COMPILE_OPTIONS "${neon_flags}")
    endif()

    add_definitions(-DHAVE_NEON=1)
endif()

# Create module library file from object library

set(library_type INTERFACE)

add_library(${TARGET_NAME} ${library_type} ${library_sources})

set_ie_threading_interface_for(${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<TARGET_PROPERTY:openvino,INTERFACE_INCLUDE_DIRECTORIES>)


# LTO

if(TARGET ${TARGET_NAME}_obj)
    set_target_properties(${TARGET_NAME}_obj ${TARGET_NAME}_s
                        PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

if(NOT library_type STREQUAL "INTERFACE")
    set_target_properties(${TARGET_NAME}
                          PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

# developer package

openvino_developer_export_targets(COMPONENT core TARGETS ${TARGET_NAME})

# install

if(BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_NAME}
            LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${OV_CPACK_COMP_CORE})
else()
    ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
endif()
