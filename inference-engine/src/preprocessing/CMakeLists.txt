# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_preproc")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

#
# Provides compilation flags for AVX512F, AVX512BW, AVX512DQ
# instructions support depending on an OS and a compiler
#
function(ie_avx512_core_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "/QxCORE-AVX512" PARENT_SCOPE)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(${flags} "/arch:AVX512" PARENT_SCOPE)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "-xCORE-AVX512" PARENT_SCOPE)
        else()
            set(${flags} "-mavx512f -mavx512bw -mavx512dq -mfma" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(set_ie_neon_optimization_flags neon_flags)
    file(GLOB NEON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/arm_neon/*.cpp)
    file(GLOB NEON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/arm_neon/*.hpp)

    list(APPEND LIBRARY_HEADERS ${NEON_HEADERS})
    list(APPEND LIBRARY_SRC ${NEON_SRC})

    set_source_files_properties(${NEON_SRC} PROPERTIES COMPILE_FLAGS "${neon_flags}")
    add_definitions(-DHAVE_NEON=1)
endfunction()

if(ENABLE_SSE42)
    file(GLOB SSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp)
    file(GLOB SSE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp)

    list(APPEND LIBRARY_HEADERS ${SSE_HEADERS})
    list(APPEND LIBRARY_SRC ${SSE_SRC})

    ie_sse42_optimization_flags(sse4_2_flags)
    set_source_files_properties(${SSE_SRC} PROPERTIES COMPILE_FLAGS "${sse4_2_flags}")
    add_definitions(-DHAVE_SSE=1)
endif()

if(ENABLE_AVX2)
    file(GLOB AVX2_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_avx2/*.cpp)
    file(GLOB AVX2_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_avx2/*.hpp)

    list(APPEND LIBRARY_HEADERS ${AVX2_HEADERS})
    list(APPEND LIBRARY_SRC ${AVX2_SRC})

    ie_avx2_optimization_flags(avx2_flags)
    set_source_files_properties(${AVX2_SRC} PROPERTIES COMPILE_FLAGS "${avx2_flags}")
    add_definitions(-DHAVE_AVX2=1)
endif()


# Workaround for GCC version 5.4 and 5.5 bugs in Debug configuration.
if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND
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
    set_source_files_properties(${AVX512_SRC} PROPERTIES COMPILE_FLAGS "${avx512_flags}")
    add_definitions(-DHAVE_AVX512=1)
endif()

if(ARM OR AARCH64)
    set(neon_flags "")
    if(WIN32 OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(neon_flags "-mfpu=neon")
        elseif(ANDROID_ABI STREQUAL "armeabi-v7a-hard with NEON")
            set(neon_flags "-march=armv7-a -mfloat-abi=hard -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -mfpu=neon")
        elseif((ANDROID_ABI STREQUAL "armeabi-v7a with NEON") OR
               (ANDROID_ABI STREQUAL "armeabi-v7a" AND
                DEFINED CMAKE_ANDROID_ARM_NEON AND CMAKE_ANDROID_ARM_NEON))
            set(neon_flags "-march=armv7-a -mfloat-abi=softfp -mfpu=neon")
        endif()
    else()
        if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
            set(neon_flags "-O2 -ftree-vectorize")
        elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7-a")
            set(neon_flags "-mfpu=neon")
        endif()
    endif()

    if(neon_flags)
        file(GLOB NEON_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/arm_neon/*.hpp)
        file(GLOB NEON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/arm_neon/*.cpp)

        list(APPEND LIBRARY_HEADERS ${NEON_HEADERS})
        list(APPEND LIBRARY_SRC ${NEON_SRC})

        set_source_files_properties(${NEON_SRC} PROPERTIES COMPILE_FLAGS "${neon_flags}")
        add_definitions(-DHAVE_NEON=1)
    endif()
endif()

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS})

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_PREPROC_PLUGIN
                                                      $<TARGET_PROPERTY:inference_engine,INTERFACE_COMPILE_DEFINITIONS>
                                                      $<TARGET_PROPERTY:fluid,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE "${IE_MAIN_SOURCE_DIR}/thirdparty/ocv"
                                                             $<TARGET_PROPERTY:fluid,INTERFACE_INCLUDE_DIRECTORIES>
                                                             $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME}_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                                                      $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>)

set_ie_threading_interface_for(${TARGET_NAME}_obj)

# Create shared library file from object library

add_library(${TARGET_NAME} SHARED
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE fluid openvino::itt
                                     PUBLIC inference_engine)

target_include_directories(${TARGET_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# Workaround to avoid warnings caused with bug in the avx512intrin.h of GCC5
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND
   (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 5.5))
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME}_s)

target_include_directories(${TARGET_NAME}_s INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
endif()

target_link_libraries(${TARGET_NAME}_s PRIVATE fluid openvino::itt
                                       INTERFACE inference_engine_s)

target_compile_definitions(${TARGET_NAME}_s INTERFACE USE_STATIC_IE)

# developer package

ie_developer_export_targets(${TARGET_NAME})

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)
