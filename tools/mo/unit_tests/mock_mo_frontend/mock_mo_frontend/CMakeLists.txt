# Copyright (C) 2018-2024 Intel Corporation
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

target_link_libraries(${TARGET_FE_NAME} PUBLIC openvino::runtime)

ov_add_clang_format_target(${TARGET_FE_NAME}_clang FOR_TARGETS ${TARGET_FE_NAME})

if(BUILD_SHARED_LIBS)
    install(TARGETS ${TARGET_FE_NAME}
            RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT tests EXCLUDE_FROM_ALL)
else()
    install(TARGETS ${TARGET_FE_NAME}
            RUNTIME DESTINATION ${OV_CPACK_PYTHONDIR}/openvino COMPONENT tests EXCLUDE_FROM_ALL
            LIBRARY DESTINATION ${OV_CPACK_PYTHONDIR}/openvino COMPONENT tests EXCLUDE_FROM_ALL)
endif()
