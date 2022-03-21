# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_FE_NAME "openvino_mock_mo_frontend")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})

# Create shared library
add_library(${TARGET_FE_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS})

target_include_directories(${TARGET_FE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_FE_NAME} PUBLIC openvino::runtime PRIVATE ngraph::builder)

add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})

if(BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_FE_NAME}
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
else()
    set(PYTHON_VERSION python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
    install(TARGETS ${TARGET_FE_NAME}
            RUNTIME DESTINATION python/${PYTHON_VERSION}/openvino COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION python/${PYTHON_VERSION}/openvino COMPONENT tests EXCLUDE_FROM_ALL)
endif()
