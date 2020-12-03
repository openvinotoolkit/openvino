# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

if (NGRAPH_INTERPRETER_ENABLE)
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
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(interpreter_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    target_link_libraries(interpreter_backend PUBLIC ngraph_backend)
    install(TARGETS interpreter_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
        RUNTIME DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
