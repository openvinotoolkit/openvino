# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/mock_frontend.cpp)
set(MOCK1_FE_NAME openvino_mock1_frontend)
add_library(${MOCK1_FE_NAME} SHARED ${SRC})

ov_add_library_version(${MOCK1_FE_NAME})

target_compile_definitions(${MOCK1_FE_NAME} PRIVATE "-DMOCK_VARIANT=\"1\"")

target_include_directories(${MOCK1_FE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${MOCK1_FE_NAME} PRIVATE openvino::frontend::common)
add_dependencies(ov_core_unit_tests ${MOCK1_FE_NAME})

ov_add_clang_format_target(${MOCK1_FE_NAME}_clang FOR_TARGETS ${MOCK1_FE_NAME})

install(TARGETS ${MOCK1_FE_NAME}
        RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
