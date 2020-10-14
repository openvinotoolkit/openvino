# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_lp_transformations")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/low_precision_transformations/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library

add_library(${TARGET_NAME} SHARED
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_API)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine openvino::itt)

target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR}
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${ENABLE_LTO})

# developer package

ie_developer_export_targets(${TARGET_NAME})

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)
