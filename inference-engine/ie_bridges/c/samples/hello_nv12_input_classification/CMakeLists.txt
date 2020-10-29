# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "hello_nv12_input_classification_c")

# create sample target

add_executable(${TARGET_NAME} main.c)

target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES})

if(COMMAND add_cpplint_target)
    add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endif()
