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

if(NOT NGRAPH_TEST_UTIL_ENABLE)
    return()
endif()

set (SRC
    all_close_f.cpp
    engine/ie_engines.cpp
    engine/interpreter_engine.cpp
    float_util.cpp
    test_tools.cpp
    test_control.cpp
    visitor.hpp
    provenance_enabler.hpp
)

add_library(ngraph_test_util STATIC ${SRC})

if(COMMAND ie_faster_build)
    ie_faster_build(ngraph_test_util
        UNITY
    )
endif()

if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(ngraph_test_util PROPERTIES
        VERSION ${NGRAPH_VERSION})
endif()
target_include_directories(ngraph_test_util PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${IE_MAIN_SOURCE_DIR}/include)
target_link_libraries(ngraph_test_util PRIVATE ngraph ngraph_backend libgtest)
