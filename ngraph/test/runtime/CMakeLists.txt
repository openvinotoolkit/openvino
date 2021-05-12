# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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
