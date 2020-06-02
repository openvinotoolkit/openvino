# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ie_docs_examples)

file(GLOB SOURCES *.cpp)

add_library(ie_docs_examples STATIC ${SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_plugin_api)

#add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
#add_clang_format_target(clang_format_${TARGET_NAME} FOR_TARGETS ${TARGET_NAME})
