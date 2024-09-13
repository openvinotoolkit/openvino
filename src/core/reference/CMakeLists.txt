# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_reference")

set(REF_IMPL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${REF_IMPL_INCLUDE_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static library
add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${PUBLIC_HEADERS})

add_library(openvino::reference ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME reference)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp")

ov_set_threading_interface_for(${TARGET_NAME})

target_compile_definitions(${TARGET_NAME} PRIVATE XBYAK_NO_OP_NAMES XBYAK64)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${REF_IMPL_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OV_CORE_DEV_API_PATH}>
    $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<BUILD_INTERFACE:$<$<TARGET_EXISTS:xbyak::xbyak>:$<TARGET_PROPERTY:xbyak::xbyak,INTERFACE_INCLUDE_DIRECTORIES>>>)

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads openvino::core::dev)

#find jpeg vars
find_path(JPEG_INCLUDE_DIR jpeglib.h REQUIRED)
set(JPEG_NAMES ${JPEG_NAMES} jpeg libjpeg)
find_library(JPEG_LIBRARIES NAMES ${JPEG_NAMES})
if(JPEG_INCLUDE_DIR AND JPEG_LIBRARIES)
  include_directories(${JPEG_INCLUDE_DIR})
  target_link_libraries(${TARGET_NAME} PRIVATE ${JPEG_LIBRARIES})
  add_compile_definitions(WITH_JPEG)
  message(STATUS "JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}")
  message(STATUS "JPEG_LIBRARIES=${JPEG_LIBRARIES}")    
else()
  message(STATUS "JPEG not found disable libjpeg supported")
endif()


ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::reference
                                    INSTALL_INCLUDE_DIRECTORIES "${REF_IMPL_INCLUDE_DIR}/")
