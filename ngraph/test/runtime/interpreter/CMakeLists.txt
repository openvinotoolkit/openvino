# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_library(interpreter_backend SHARED int_backend.cpp int_executable.cpp evaluates_map.cpp)

ie_faster_build(interpreter_backend UNITY)

ie_add_vs_version_file(NAME interpreter_backend
                       FILEDESCRIPTION "nGraph interpreter backend library")

ie_add_api_validator_post_build_step(TARGET interpreter_backend)

target_compile_definitions(interpreter_backend PRIVATE INTERPRETER_BACKEND_EXPORTS)
target_link_libraries(interpreter_backend PUBLIC ngraph_backend PRIVATE ngraph_reference)

install(TARGETS interpreter_backend
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS interpreter_backend
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()
