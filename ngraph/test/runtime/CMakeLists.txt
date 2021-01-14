# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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
    executable.cpp
    executable.hpp
    performance_counter.hpp
    dynamic/dynamic_backend.cpp
    dynamic/dynamic_backend.hpp
    op/avg_pool.cpp
    op/avg_pool.hpp
    op/convolution.cpp
    op/convolution.hpp
    op/group_conv.cpp
    op/group_conv.hpp
    pass/dyn_elimination.cpp
    pass/dyn_elimination.hpp
    pass/fused_op_decomposition.cpp
    pass/fused_op_decomposition.hpp
    pass/implicit_broadcast_elimination.cpp
    pass/implicit_broadcast_elimination.hpp
    pass/liveness.cpp
    pass/liveness.hpp
    pass/opset0_downgrade.cpp
    pass/opset0_downgrade.hpp
    pass/opset1_downgrade.cpp
    pass/opset1_downgrade.hpp
    pass/opset1_upgrade.cpp
    pass/opset1_upgrade.hpp
    pass/shape_relevance.cpp
    pass/shape_relevance.hpp
    )

disable_deprecated_warnings()

add_library(ngraph_backend SHARED ${SRC})

if(COMMAND ie_faster_build)
    ie_faster_build(ngraph_backend
        UNITY
    )
endif()

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ngraph_backend
                           FILEDESCRIPTION "nGraph backend library")
endif()

target_compile_definitions(ngraph_backend
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(ngraph_backend PUBLIC ngraph
                                            ngraph::builder
                                            ngraph::reference)
if (NOT WIN32)
    target_link_libraries(ngraph_backend PRIVATE ${CMAKE_DL_LIBS})
endif()
target_include_directories(ngraph_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(interpreter)
add_subdirectory(ie)
