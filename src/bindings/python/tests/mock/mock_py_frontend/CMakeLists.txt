# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_FRONTEND_NAME "mock_py")
set(TARGET_NAME "${FRONTEND_NAME_PREFIX}${OV_FRONTEND_NAME}${FRONTEND_NAME_SUFFIX}")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

# Build as shared library because of static members.
# If build as static library then these members will be duplicated e.g. per pybind shared library and test will fail.
# Using static methods (statistics) from pybind_mock_frontend access different data then
# mock fornt end created by pyopenvino.
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS})
add_library(openvino::frontend::mock_py ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(${TARGET_NAME} PUBLIC
    $<$<BOOL:${ENABLE_OV_ONNX_FRONTEND}>:ENABLE_OV_ONNX_FRONTEND>
    $<$<BOOL:${ENABLE_OV_TF_FRONTEND}>:ENABLE_OV_TF_FRONTEND>
    $<$<BOOL:${ENABLE_OV_PADDLE_FRONTEND}>:ENABLE_OV_PADDLE_FRONTEND>)

target_link_libraries(${TARGET_NAME} PUBLIC
    openvino::runtime
    $<$<BOOL:${ENABLE_OV_ONNX_FRONTEND}>:openvino::frontend::onnx>
    $<$<BOOL:${ENABLE_OV_TF_FRONTEND}>:openvino::frontend::tensorflow>
    $<$<BOOL:${ENABLE_OV_PADDLE_FRONTEND}>:openvino::frontend::paddle>)

ov_add_library_version(${TARGET_NAME})
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
    LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
)
