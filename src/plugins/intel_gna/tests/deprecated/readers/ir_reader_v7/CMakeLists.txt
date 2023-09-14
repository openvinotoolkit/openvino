# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "inference_engine_ir_v7_reader")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/*.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})

# Create module library

if(BUILD_SHARED_LIBS)
    set(library_type MODULE)
else()
    set(library_type STATIC)
endif()

add_library(${TARGET_NAME} ${library_type} ${LIBRARY_SRC})

ov_build_target_faster(${TARGET_NAME}
    UNITY
)

ov_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Inference Engine IR v7 reader plugin")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/")

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::dev
                                             inference_engine_legacy openvino::pugixml openvino::itt)

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)
endif()

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# code style

ov_add_clang_format_target(${TARGET_NAME}_clang_format FOR_TARGETS ${TARGET_NAME})

# Install rules

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
