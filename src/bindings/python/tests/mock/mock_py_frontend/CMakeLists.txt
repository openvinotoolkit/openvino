# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FE_NAME "mock_py")
set(TARGET_FE_NAME "${FRONTEND_NAME_PREFIX}${FE_NAME}${FRONTEND_NAME_SUFFIX}")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

add_library(${TARGET_FE_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS})
ov_add_library_version(${TARGET_FE_NAME})

target_include_directories(${TARGET_FE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(${TARGET_FE_NAME} PRIVATE
    IMPLEMENT_OPENVINO_API
    $<$<BOOL:${ENABLE_OV_ONNX_FRONTEND}>:ENABLE_OV_ONNX_FRONTEND>
    $<$<BOOL:${ENABLE_OV_TF_FRONTEND}>:ENABLE_OV_TF_FRONTEND>
    $<$<BOOL:${ENABLE_OV_PADDLE_FRONTEND}>:ENABLE_OV_PADDLE_FRONTEND>)

target_link_libraries(${TARGET_FE_NAME} PUBLIC
    openvino::runtime
    $<$<BOOL:${ENABLE_OV_ONNX_FRONTEND}>:openvino::frontend::onnx>
    $<$<BOOL:${ENABLE_OV_TF_FRONTEND}>:openvino::frontend::tensorflow>
    $<$<BOOL:${ENABLE_OV_PADDLE_FRONTEND}>:openvino::frontend::paddle>)

add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})

install(TARGETS ${TARGET_FE_NAME}
    RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT tests EXCLUDE_FROM_ALL
    LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT tests EXCLUDE_FROM_ALL)
