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
set (SRC
    backend.cpp
    backend.hpp
    backend_manager.cpp
    backend_manager.hpp
    cache.cpp
    cache.hpp
    opset0_downgrade.cpp
    opset0_downgrade.hpp
    executable.cpp
    executable.hpp
    performance_counter.hpp
    dynamic/dynamic_backend.cpp
    dynamic/dynamic_backend.hpp
    op/avg_pool.cpp
    op/avg_pool.hpp
    )

add_library(ngraph_backend SHARED ${SRC})
target_compile_definitions(ngraph_backend
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(ngraph_backend PUBLIC ngraph)
if (NOT WIN32)
    target_link_libraries(ngraph_backend PRIVATE dl)
endif()
target_compile_definitions(ngraph_backend PRIVATE BACKEND_DLL_EXPORTS)
target_include_directories(ngraph_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(interpreter)

if (NGRAPH_IE_ENABLE)
    add_subdirectory(ie)
endif()
