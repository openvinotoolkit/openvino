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

# Create object library

if(ENABLE_GAPI_PREPROCESSING)
    add_library(${TARGET_NAME}_obj OBJECT
                ${LIBRARY_SRC}
                ${LIBRARY_HEADERS})
    ov_build_target_faster(${TARGET_NAME}_obj UNITY)

    target_compile_definitions(${TARGET_NAME}_obj PRIVATE
        IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        $<TARGET_PROPERTY:openvino::runtime,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:fluid,INTERFACE_COMPILE_DEFINITIONS>
        ENABLE_GAPI_PREPROCESSING)

    target_include_directories(${TARGET_NAME}_obj SYSTEM PRIVATE
        $<TARGET_PROPERTY:ocv_hal,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:fluid,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        $<TARGET_PROPERTY:openvino::util,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>)

    ov_set_threading_interface_for(${TARGET_NAME}_obj)

    add_cpplint_target(${TARGET_NAME}_obj_cpplint FOR_TARGETS ${TARGET_NAME}_obj)

    set(library_sources $<TARGET_OBJECTS:${TARGET_NAME}_obj>)
endif()

# Create module library file from object library

if(ENABLE_GAPI_PREPROCESSING)
    if(BUILD_SHARED_LIBS)
        set(library_type MODULE)
    else()
        set(library_type STATIC)
    endif()
else()
    set(library_type INTERFACE)
endif()

add_library(${TARGET_NAME} ${library_type} ${library_sources} ${CMAKE_CURRENT_SOURCE_DIR}/ie_preprocess_dummy.cpp)

ov_set_threading_interface_for(${TARGET_NAME})

if(ENABLE_GAPI_PREPROCESSING)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_GAPI_PREPROCESSING)
    target_link_libraries(${TARGET_NAME} PRIVATE fluid openvino::itt openvino::util)

    if(BUILD_SHARED_LIBS)
        target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
        # make plugin to depend on preprocessing library
        foreach(plugin IN LISTS PLUGIN_FILES)
            string(REPLACE ":" ";" plugin "${plugin}")
            list(REMOVE_AT plugin 0)
            add_dependencies(${plugin} openvino_gapi_preproc)
        endforeach()
        add_dependencies(ov_plugins openvino_gapi_preproc)
    else()
        # for static linkage the dependencies are in opposite order
        target_link_libraries(openvino PRIVATE ${TARGET_NAME})
    endif()

    # Workaround to avoid warnings caused with bug in the avx512intrin.h of GCC5
    if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND
       (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 5.5))
        set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
    endif()

    if(WIN32 AND NOT library_type STREQUAL "INTERFACE")
        set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
    endif()

    # must be called after all target_link_libraries
    ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

    ov_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "OpenVINO Preprocessing plugin")
endif()

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<TARGET_PROPERTY:openvino,INTERFACE_INCLUDE_DIRECTORIES>)

# Static library used for unit tests which are always built

if(ENABLE_GAPI_PREPROCESSING)
    add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
                $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

    ov_set_threading_interface_for(${TARGET_NAME}_s)

    target_include_directories(${TARGET_NAME}_s INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

    if(WIN32)
        set_target_properties(${TARGET_NAME}_s PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_s)
    endif()

    target_link_libraries(${TARGET_NAME}_s PRIVATE fluid openvino::itt openvino::runtime)

    target_compile_definitions(${TARGET_NAME}_s INTERFACE USE_STATIC_IE)

    set_target_properties(${TARGET_NAME}_s PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

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
