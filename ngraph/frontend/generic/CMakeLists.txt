# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create shared library
add_library(frontend_manager SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(ngraph::frontend_manager ALIAS frontend_manager)

target_link_libraries(frontend_manager PRIVATE ${CMAKE_DL_LIBS} ngraph)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME frontend_manager
                           FILEDESCRIPTION "Manager of OpenVINO nGraph Front Ends")
endif()

set(FRONTEND_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend/generic")
target_include_directories(frontend_manager SYSTEM PUBLIC $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>)
target_include_directories(frontend_manager SYSTEM PRIVATE ${NGRAPH_INCLUDE_PATH} ${FRONTEND_INCLUDE_DIR})

target_include_directories(frontend_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(frontend_manager PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-invalid-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated PUBLIC -Wno-undefined-func-template)
endif()

# TODO: uncomment to include frontend_manager to installation package
#install(TARGETS frontend_manager EXPORT ngraphTargets
#        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
#        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
#        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
#
#install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/frontend_manager
#    DESTINATION ${FRONTEND_INSTALL_INCLUDE}
#    COMPONENT ngraph
#    FILES_MATCHING
#        PATTERN "*.hpp"
#        PATTERN "*.h"
#)
#
#if (NGRAPH_EXPORT_TARGETS_ENABLE)
#    export(TARGETS frontend_manager NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")
#endif()
