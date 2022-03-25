# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
if (ENABLE_OV_PADDLE_FRONTEND)
    add_subdirectory(paddle)
endif()

if (ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/mock_frontend.cpp)
set(MOCK1_FE_NAME openvino_mock1_frontend)
add_library(${MOCK1_FE_NAME} SHARED EXCLUDE_FROM_ALL ${SRC})

target_compile_definitions(${MOCK1_FE_NAME} PRIVATE "-DMOCK_VARIANT=\"1\"")

target_include_directories(${MOCK1_FE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${MOCK1_FE_NAME} PRIVATE frontend_common)
add_dependencies(ov_core_unit_tests ${MOCK1_FE_NAME})

add_clang_format_target(${MOCK1_FE_NAME}_clang FOR_TARGETS ${MOCK1_FE_NAME})

if(BUILD_SHARED_LIBS)
    install(TARGETS ${MOCK1_FE_NAME}
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL)
else()
    install(TARGETS ${MOCK1_FE_NAME}
            RUNTIME DESTINATION tests COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL)
endif()
