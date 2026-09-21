# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME tp_benchmark)

add_executable(${TARGET_NAME} tp_benchmark.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

if(TARGET LevelZero::LevelZero)
    target_link_options(${TARGET_NAME} PRIVATE "LINKER:--no-as-needed")
    target_link_libraries(${TARGET_NAME} PRIVATE LevelZero::LevelZero)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")
