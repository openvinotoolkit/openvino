# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_snippets")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (ENABLE_DEBUG_CAPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_CAPS")
endif()

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/snippets/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

add_library(${TARGET_NAME}_static_ie STATIC EXCLUDE_FROM_ALL
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

target_compile_definitions(${TARGET_NAME}_static_ie PRIVATE USE_STATIC_IE)

macro(finalize_libs)
    foreach(target IN ITEMS ${ARGN})
        ie_faster_build(${target}
            UNITY
        )

        target_link_libraries(${target} PUBLIC openvino::runtime
                                        PRIVATE ngraph_reference openvino::runtime::dev)

        target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
                                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

        add_cpplint_target(${target}_cpplint FOR_TARGETS ${target})

        ie_mark_target_as_cc(${target})

        # LTO
        set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

        # install
        ov_install_static_lib(${target} core)
    endforeach()
endmacro()

finalize_libs(${TARGET_NAME} ${TARGET_NAME}_static_ie)
