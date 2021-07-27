# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_library(interpreter_backend SHARED int_backend.cpp int_executable.cpp evaluates_map.cpp)

if(COMMAND ie_faster_build)
    ie_faster_build(interpreter_backend
        UNITY
    )
endif()

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME interpreter_backend
                            FILEDESCRIPTION "nGraph interpreter backend library")
endif()

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME interpreter_backend
                            FILEDESCRIPTION "nGraph interpreter backend library")
endif()

target_compile_definitions(interpreter_backend PRIVATE INTERPRETER_BACKEND_EXPORTS)
target_link_libraries(interpreter_backend PUBLIC ngraph_backend)

install(TARGETS interpreter_backend
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT tests EXCLUDE_FROM_ALL)
