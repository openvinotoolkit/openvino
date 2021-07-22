# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "offline_transformations_api")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/offline_transformations)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/offline_transformations)
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/offline_transformations)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${PYTHON_BRIDGE_OUTPUT_DIRECTORY}/offline_transformations)

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/offline_transformations_api_impl_defs.pxd
            ${CMAKE_CURRENT_SOURCE_DIR}/offline_transformations_api.pyx
            ${CMAKE_CURRENT_SOURCE_DIR}/offline_transformations_api_impl.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/offline_transformations_api_impl.cpp)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/offline_transformations_api.pyx
                            PROPERTIES CYTHON_IS_CXX ON)

# create target

cython_add_module(${TARGET_NAME} ${SOURCES})
add_dependencies(${TARGET_NAME} ie_api)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "Offline Transformatoins Python library")
endif()

if(TARGET offline_transformations)
    list(APPEND InferenceEngine_LIBRARIES offline_transformations)
else()
    list(APPEND InferenceEngine_LIBRARIES IE::offline_transformations)
endif()

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../inference_engine")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

# Compatibility with python 2.7 which has deprecated "register" specifier
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE "-Wno-error=register")
endif()

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                        EXCLUDE_PATTERNS ".*\\.cxx;.*\\.pxd;.*\\.pyx")

# perform copy
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_BRIDGE_SRC_ROOT}/src/openvino/offline_transformations/__init__.py ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
)

# install

# TODO: use ${PYTHON_VERSION}_dev component below
# ie_cpack_add_component(${PYTHON_VERSION}_dev DEPENDS ${PYTHON_COMPONENT})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION python/${PYTHON_VERSION}/openvino/offline_transformations COMPONENT ${PYTHON_COMPONENT}
        LIBRARY DESTINATION python/${PYTHON_VERSION}/openvino/offline_transformations COMPONENT ${PYTHON_COMPONENT})

install(PROGRAMS __init__.py
        DESTINATION python/${PYTHON_VERSION}/openvino/offline_transformations
        COMPONENT ${PYTHON_COMPONENT})
