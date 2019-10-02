# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_preproc")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if((NOT DEFINED ENABLE_SSE42) OR ENABLE_SSE42)
    file(GLOB LIBRARY_SRC ${LIBRARY_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.cpp)
    file(GLOB LIBRARY_HEADERS ${LIBRARY_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/*.hpp)
    if (WIN32)
        set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.cpp" PROPERTIES COMPILE_FLAGS /arch:SSE2)
    else()
        set_source_files_properties(
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_data_sse42.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.cpp" PROPERTIES COMPILE_FLAGS -msse4.2)
    endif()

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpu_x86_sse42)
    add_definitions(-DHAVE_SSE=1)
endif()

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS})

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_PREPROC_PLUGIN
                                                      $<TARGET_PROPERTY:inference_engine,INTERFACE_COMPILE_DEFINITIONS>
                                                      $<TARGET_PROPERTY:fluid,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
                                                             $<TARGET_PROPERTY:fluid,INTERFACE_INCLUDE_DIRECTORIES>
                                                             "${IE_MAIN_SOURCE_DIR}/thirdparty/ocv"
                                                             "${IE_MAIN_SOURCE_DIR}/src/inference_engine")

if(ENABLE_PROFILING_ITT AND INTEL_ITT_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_COMPILE_DEFINITIONS>)
    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE $<TARGET_PROPERTY:ittnotify,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

target_include_directories(${TARGET_NAME}_obj PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    # To disable min/max macro in windows.h
    target_compile_definitions(${TARGET_NAME}_obj PRIVATE NOMINMAX)
endif()

set_ie_threading_interface_for(${TARGET_NAME}_obj)

# Create shared library file from object library

add_library(${TARGET_NAME} SHARED
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE fluid PUBLIC inference_engine ${INTEL_ITT_LIBS})

target_include_directories(${TARGET_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME}_s)

target_include_directories(${TARGET_NAME}_s INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
endif()

target_link_libraries(${TARGET_NAME}_s PRIVATE fluid INTERFACE inference_engine_s ${INTEL_ITT_LIBS})

target_compile_definitions(${TARGET_NAME}_s INTERFACE USE_STATIC_IE)

# developer package

export(TARGETS ${TARGET_NAME} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
