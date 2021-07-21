# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_manager")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Static library

add_library(${TARGET_NAME}_static STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(ngraph::${TARGET_NAME}::static ALIAS ${TARGET_NAME}_static)
target_link_libraries(${TARGET_NAME}_static PRIVATE ${CMAKE_DL_LIBS} PUBLIC ngraph)
target_include_directories(${TARGET_NAME}_static PUBLIC ${FRONTEND_INCLUDE_DIR})
target_include_directories(${TARGET_NAME}_static PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(${TARGET_NAME}_static PUBLIC USE_STATIC_FRONTEND_MANAGER)


# Shared library - need to recompile object files to export necessary symbols

add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(ngraph::${TARGET_NAME} ALIAS ${TARGET_NAME})
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS} PUBLIC ngraph)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "Manager of OpenVINO nGraph Frontends")
endif()

# Installation rules for shared version only

install(TARGETS ${TARGET_NAME} EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)

install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/frontend_manager
    DESTINATION ${FRONTEND_INSTALL_INCLUDE}
    COMPONENT ngraph_dev
    FILES_MATCHING PATTERN "*.hpp")

export(TARGETS ${TARGET_NAME} NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
