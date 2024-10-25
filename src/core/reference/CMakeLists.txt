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

if(ENABLE_AVX2)
    ov_avx2_optimization_flags(avx2_flags)

    set(OV_REFERENCE_X86_AVX2_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/op/convert_x86_intrinsics.cpp
    )
    set_source_files_properties(${OV_REFERENCE_X86_AVX2_SRC} PROPERTIES COMPILE_OPTIONS "${avx2_flags}"
                                                                        SKIP_UNITY_BUILD_INCLUSION ON
                                                                        SKIP_PRECOMPILE_HEADERS ON)
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_AVX2)
endif()

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

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads openvino::core::dev)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::reference
                                    INSTALL_INCLUDE_DIRECTORIES "${REF_IMPL_INCLUDE_DIR}/")
