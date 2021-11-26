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
    pass/dyn_elimination.cpp
    pass/dyn_elimination.hpp
    pass/shape_relevance.cpp
    pass/shape_relevance.hpp
    )

add_library(ngraph_backend SHARED ${SRC})

ie_faster_build(ngraph_backend UNITY)

ie_add_vs_version_file(NAME ngraph_backend
                       FILEDESCRIPTION "nGraph backend library")

ie_add_api_validator_post_build_step(TARGET ngraph_backend)

target_compile_definitions(ngraph_backend
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(ngraph_backend PUBLIC ngraph ngraph::builder ngraph::reference
                                     PRIVATE ${CMAKE_DL_LIBS} openvino::util)

target_include_directories(ngraph_backend PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

install(TARGETS ngraph_backend
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ngraph_backend
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endif()

add_subdirectory(interpreter)
add_subdirectory(ie)

file(GLOB_RECURSE all_backends_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(ngraph_backend_clang FOR_SOURCES ${all_backends_src})
