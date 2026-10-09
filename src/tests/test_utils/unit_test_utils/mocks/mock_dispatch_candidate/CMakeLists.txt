# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Two independent candidate module libraries built from one source, tagged "A"/"B".
# Distinct .so files give each its own globals so the dispatch tests can script them
# separately (registering the same .so twice would share a single dlopen handle).
foreach(SUFFIX a b)
    string(TOUPPER ${SUFFIX} TAG)
    set(TARGET_NAME "mock_dispatch_candidate_${SUFFIX}")

    add_library(${TARGET_NAME} MODULE ${CMAKE_CURRENT_SOURCE_DIR}/mock_dispatch_candidate.cpp)

    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::dev)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        IMPLEMENT_OPENVINO_RUNTIME_PLUGIN
        MOCK_CANDIDATE_TAG="${TAG}")

    install(TARGETS ${TARGET_NAME}
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endforeach()
