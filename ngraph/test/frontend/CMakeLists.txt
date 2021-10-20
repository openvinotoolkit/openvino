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
add_library(mock1_ngraph_frontend SHARED ${SRC})

target_compile_definitions(mock1_ngraph_frontend PRIVATE "-DMOCK_VARIANT=\"1\"")

target_include_directories(mock1_ngraph_frontend PRIVATE ".")

target_link_libraries(mock1_ngraph_frontend PRIVATE frontend_manager)
add_dependencies(unit-test mock1_ngraph_frontend)

add_clang_format_target(mock1_ngraph_frontend_clang FOR_TARGETS mock1_ngraph_frontend)

install(TARGETS mock1_ngraph_frontend
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
