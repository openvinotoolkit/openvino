# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_decompositions")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# OBJECT library shared by frontends and transformation tests.
# Provides reference decompositions guaranteed to be matched by the corresponding
# fusion transformations (RMSFusion, RoPEFusion, ...). Built as an OBJECT library
# so that consumers absorb its object files directly and there is no separate
# install/export target to manage.
add_library(${TARGET_NAME}_obj OBJECT ${LIBRARY_SRC} ${PUBLIC_HEADERS})

set_target_properties(${TARGET_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET_NAME}_obj PUBLIC
    $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

# Link against openvino::core::dev (development headers INTERFACE target),
# not openvino::runtime: the OBJECT library only needs OpenVINO core/op headers
# at compile time. Depending on the runtime shared library here breaks
# cross-compilation and static builds where consumers (frontends) absorb the
# object files and bring in runtime symbols via their own link line.
target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::core::dev)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME}_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

# INTERFACE wrapper that propagates both the headers and the object files to
# any target that links against it. This keeps `openvino::decompositions` as
# the single, friendly link target for downstream callers.
add_library(${TARGET_NAME} INTERFACE)
add_library(openvino::decompositions ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

target_link_libraries(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:$<TARGET_OBJECTS:${TARGET_NAME}_obj>>
    openvino::runtime)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}_obj)

ov_build_target_faster(${TARGET_NAME}_obj PCH)

# In static builds, frontends (e.g. openvino_onnx_frontend, openvino_pytorch_frontend)
# become STATIC libraries and propagate their PRIVATE link dependencies via
# INTERFACE_LINK_LIBRARIES. As a result, the exported OpenVINOTargets references
# `openvino_decompositions` and CMake requires the target itself to be in the same
# export set. Install/export the INTERFACE wrapper unconditionally; since its
# include dirs and object files are wrapped in $<BUILD_INTERFACE:...>, the
# installed config only propagates `openvino::runtime` (already exported), and no
# binary artifacts or headers are installed.
install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets)
export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
