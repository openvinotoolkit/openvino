# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "vpu_graph_transformer")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

add_library(${TARGET_NAME} STATIC ${SOURCES})

set_ie_threading_interface_for(${TARGET_NAME})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
# TODO: enable some day and fix all warnings
#     target_compile_options(${TARGET_NAME} PRIVATE "-Wall")
    target_compile_options(${TARGET_NAME} PRIVATE "-Werror=unused-variable")
    target_compile_options(${TARGET_NAME} PRIVATE "-Werror=unused-function")
    target_compile_options(${TARGET_NAME} PRIVATE "-Werror=strict-aliasing")
endif()

target_include_directories(${TARGET_NAME}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(${TARGET_NAME}
    SYSTEM PUBLIC
        "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src"
        "${IE_MAIN_SOURCE_DIR}/include"
        "${IE_MAIN_SOURCE_DIR}/src/inference_engine"
        "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include")

target_link_libraries(${TARGET_NAME} PUBLIC pugixml vpu_common_lib)

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)

    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
add_cppcheck(${TARGET_NAME})

# unit tests support for graph transformer
if (WIN32)
    add_library(${TARGET_NAME}_test_static STATIC ${SOURCES})

    set_ie_threading_interface_for(${TARGET_NAME}_test_static)

    # static linkage to inference_engine library
    target_compile_definitions(${TARGET_NAME}_test_static PUBLIC USE_STATIC_IE PRIVATE NOMINMAX)

    get_target_property(target_includes ${TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${TARGET_NAME}_test_static SYSTEM PUBLIC ${target_includes})

    set_target_properties(${TARGET_NAME}_test_static PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_test_static)

    target_link_libraries(${TARGET_NAME}_test_static PUBLIC vpu_common_lib)
else()
    add_library(${TARGET_NAME}_test_static ALIAS ${TARGET_NAME})
endif()

#
# developer package
#

export(TARGETS ${TARGET_NAME} NAMESPACE IE::
        APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
