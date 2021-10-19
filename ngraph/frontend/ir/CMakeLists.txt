# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ir_ngraph_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create library
add_library(${TARGET_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:frontend_manager::static,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "FrontEnd to load and convert IR file format")
endif()

target_link_libraries(${TARGET_NAME} PRIVATE frontend_manager::static
        ngraph::builder inference_engine_transformations
        inference_engine pugixml::static inference_engine_plugin_api)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                        EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ngraph
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ngraph
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ngraph)
