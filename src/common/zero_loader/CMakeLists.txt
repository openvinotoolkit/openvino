# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_zero_loader)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})
add_library(openvino::zero_loader ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PUBLIC LevelZero::Headers PRIVATE openvino::runtime openvino::util)
target_include_directories(${TARGET_NAME} 
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ov_developer_package_export_targets(TARGET level_zero_headers)
ov_install_static_lib(level_zero_headers ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
    INSTALL_INCLUDE_DIRECTORIES 
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

if(TARGET ze_loader)
    add_dependencies(${TARGET_NAME} ze_loader)
endif()
