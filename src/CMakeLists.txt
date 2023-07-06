# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions(-DIN_OV_COMPONENT)
add_definitions(-DPROJECT_ROOT_DIR="${OpenVINO_SOURCE_DIR}")

cmake_path(NATIVE_PATH OpenVINO_SOURCE_DIR NATIVE_PROJECT_ROOT_DIR)
string(LENGTH "${NATIVE_PROJECT_ROOT_DIR}/" PROJECT_ROOT_DIR_LENGTH)

add_compile_definitions(

    # Compiler not support to trim __FILE__ to have got relative path, use offset instead
    $<$<NOT:$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU,MSVC>>:PROJECT_ROOT_DIR_LENGTH=${PROJECT_ROOT_DIR_LENGTH}>)

add_compile_options(

    # Enable __FILE__ trimming by compiler for:
    # AppleClang, Clang, GNU
    $<$<COMPILE_LANG_AND_ID:C,AppleClang,Clang,GNU>:-ffile-prefix-map=${NATIVE_PROJECT_ROOT_DIR}/=>
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-ffile-prefix-map=${NATIVE_PROJECT_ROOT_DIR}/=>

    # MSVC
    $<$<COMPILE_LANG_AND_ID:C,MSVC>:/d1trimfile:${NATIVE_PROJECT_ROOT_DIR}\\>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/d1trimfile:${NATIVE_PROJECT_ROOT_DIR}\\>)

include(cmake/install_tbb.cmake)

# CC library should be registered before other cc targets
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(frontends)
# TODO: remove ngraph/ngraph.hpp usage
if(ENABLE_TESTS)
    add_subdirectory(core/tests)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(plugins)
add_subdirectory(inference)
include(cmake/openvino.cmake)
# preprocessing has dependency on `openvino` for static build
add_subdirectory(common/preprocessing)
add_subdirectory(bindings)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
