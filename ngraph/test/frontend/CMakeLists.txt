# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
add_subdirectory(shared)

if (NGRAPH_PDPD_FRONTEND_ENABLE)
    add_subdirectory(paddlepaddle)
endif()

if (NGRAPH_ONNX_FRONTEND_ENABLE)
    add_subdirectory(onnx)
endif()

if (NGRAPH_TF_FRONTEND_ENABLE)
    add_subdirectory(tensorflow)
endif()

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/mock_frontend.cpp)
set(MOCK1_FE_NAME mock1_ov_frontend)
add_library(${MOCK1_FE_NAME} SHARED ${SRC})

target_compile_definitions(${MOCK1_FE_NAME} PRIVATE "-DMOCK_VARIANT=\"1\"")

target_include_directories(${MOCK1_FE_NAME} PRIVATE ".")

target_link_libraries(${MOCK1_FE_NAME} PRIVATE frontend_manager)
add_dependencies(unit-test ${MOCK1_FE_NAME})

add_clang_format_target(${MOCK1_FE_NAME}_clang FOR_TARGETS ${MOCK1_FE_NAME})

if(BUILD_SHARED_LIBS OR NOT WIN32)
    install(TARGETS ${MOCK1_FE_NAME}
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
else()
    install(TARGETS ${MOCK1_FE_NAME}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()
