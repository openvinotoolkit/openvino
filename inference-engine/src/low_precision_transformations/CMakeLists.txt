# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "inference_engine_lp_transformations")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/low_precision_transformations/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create object library

add_library(${TARGET_NAME}_obj OBJECT
            ${LIBRARY_SRC}
            ${LIBRARY_HEADERS})

set_ie_threading_interface_for(${TARGET_NAME}_obj)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE IMPLEMENT_INFERENCE_ENGINE_API
                                                      $<TARGET_PROPERTY:inference_engine,INTERFACE_COMPILE_DEFINITIONS>
                                                      $<TARGET_PROPERTY:openvino::itt,INTERFACE_COMPILE_DEFINITIONS>)

target_include_directories(${TARGET_NAME}_obj PRIVATE ${PUBLIC_HEADERS_DIR}
                                                      $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
                                                      $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME}_obj)

# Create shared library file from object library

file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)

add_library(${TARGET_NAME} SHARED
            # according to https://cmake.org/cmake/help/latest/command/add_library.html#id4
            # Some native build systems (such as Xcode) may not like targets that have only
            # object files, so consider adding at least one real source file to any target that
            # references $<TARGET_OBJECTS:objlib>.
            ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
            $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE ${PUBLIC_HEADERS_DIR}
                                                    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine openvino::itt)

# Static library used for unit tests which are always built

add_library(${TARGET_NAME}_s STATIC $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

set_ie_threading_interface_for(${TARGET_NAME}_s)

target_compile_definitions(${TARGET_NAME}_s INTERFACE USE_STATIC_IE)

target_include_directories(${TARGET_NAME}_s INTERFACE ${PUBLIC_HEADERS_DIR}
                                                      $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME}_s PUBLIC inference_engine_s openvino::itt)

# developer package

ie_developer_export_targets(${TARGET_NAME} ${TARGET_NAME}_s)

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)
