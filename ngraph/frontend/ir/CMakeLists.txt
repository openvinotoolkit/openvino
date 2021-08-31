# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ir_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} SHARED  ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:ngraph::frontend_manager::static,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "FrontEnd to load and convert IR file format")
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ngraph::frontend_manager::static
                                     PRIVATE ngraph::builder inference_engine_transformations
        inference_engine pugixml::static inference_engine_plugin_api)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                        EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

install(TARGETS ${TARGET_NAME} EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/ir_frontend
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ngraph_dev
        FILES_MATCHING PATTERN "*.hpp"
)

export(TARGETS ${TARGET_NAME} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
