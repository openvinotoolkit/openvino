# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_tensorflow_common")
get_filename_component(root_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

file(GLOB_RECURSE LIBRARY_SRC ${root_dir}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${root_dir}/include/*.hpp)

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS})
add_library(openvino::frontend::tensorflow_common ALIAS ${TARGET_NAME})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENVINO_STATIC_LIBRARY)
endif()

# remove -Wmissing-declarations warning, because of frontends implementation specific
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-missing-declarations)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_link_libraries(${TARGET_NAME} PRIVATE openvino::util openvino::frontend::common_translators
                                     PUBLIC openvino::core::dev)

target_include_directories(${TARGET_NAME}
    PUBLIC $<BUILD_INTERFACE:${root_dir}/include>
    PRIVATE ${root_dir}/src)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES "${root_dir}/include"
                                       "${root_dir}/src"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:${TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                        $<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>)
