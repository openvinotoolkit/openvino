# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_tp_gpu_func_tests)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::runtime
    common_test_utils
    gtest
    gtest_main)

# The GPU plugin dlopens libze_loader by soname.  The test itself calls no L0
# symbol, so --as-needed would drop the dependency; forcing it keeps the loader
# in the process image, where dlopen resolves it by soname without a search.
# That also puts the loader's build directory on the test's RPATH.
if(TARGET LevelZero::LevelZero)
    target_link_options(${TARGET_NAME} PRIVATE "LINKER:--no-as-needed")
    target_link_libraries(${TARGET_NAME} PRIVATE LevelZero::LevelZero)
endif()

# The plugins are loaded at run time, but must exist before the test runs.
add_dependencies(${TARGET_NAME} openvino_tp_gpu_plugin openvino_intel_gpu_plugin)

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "TP_GPU")
