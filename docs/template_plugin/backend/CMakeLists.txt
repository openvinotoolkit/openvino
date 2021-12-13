# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (SRC
    backend.cpp
    backend.hpp
    cache.cpp
    cache.hpp
    executable.cpp
    executable.hpp
    performance_counter.hpp
    pass/dyn_elimination.cpp
    pass/dyn_elimination.hpp
    pass/shape_relevance.cpp
    pass/shape_relevance.hpp
    int_backend.cpp int_executable.cpp evaluates_map.cpp
    )

add_library(interpreter_backend STATIC EXCLUDE_FROM_ALL ${SRC})

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
endif()

ie_faster_build(interpreter_backend UNITY)

target_compile_definitions(interpreter_backend
    PRIVATE
        SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        SHARED_LIB_SUFFIX="${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
target_link_libraries(interpreter_backend PRIVATE ngraph::builder ngraph::reference openvino::util)

target_include_directories(interpreter_backend PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

file(GLOB_RECURSE all_backends_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
add_clang_format_target(interpreter_backend_clang FOR_SOURCES ${all_backends_src})


# developer package
openvino_developer_export_targets(COMPONENT core TARGETS interpreter_backend)

install(TARGETS interpreter_backend
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL
        ARCHIVE DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL)
if(NOT BUILD_SHARED_LIBS)
    install(TARGETS interpreter_backend
            RUNTIME DESTINATION tests COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL
            ARCHIVE DESTINATION tests COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL
            LIBRARY DESTINATION tests COMPONENT tests OPTIONAL EXCLUDE_FROM_ALL)
endif()
