# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

function(add_common_target TARGET_NAME STATIC_IE)
    if(STATIC_IE)
        set(exclude_from_all EXCLUDE_FROM_ALL)
    endif()

    add_library(${TARGET_NAME} STATIC ${exclude_from_all} ${SOURCES})

    ie_faster_build(${TARGET_NAME}
        UNITY
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # TODO: enable some day and fix all warnings
#        target_compile_options(${TARGET_NAME} PRIVATE "-Wall")
        target_compile_options(${TARGET_NAME} PRIVATE "-Werror=unused-function")
        target_compile_options(${TARGET_NAME} PRIVATE "-Werror=strict-aliasing")
    endif()

    if(STATIC_IE)
        # static linkage to inference_engine library
        target_compile_definitions(${TARGET_NAME}
            PUBLIC
                USE_STATIC_IE)
        set_target_properties(${TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

    target_include_directories(${TARGET_NAME}
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include")

    target_include_directories(${TARGET_NAME} SYSTEM PUBLIC
            $<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

    if(WIN32)
        set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
    endif()

    if(NOT STATIC_IE)
        add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME} CUSTOM_FILTERS "+runtime/explicit")
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

    openvino_developer_export_targets(COMPONENT inference_engine_vpu TARGETS ${TARGET_NAME})

    target_link_libraries(${TARGET_NAME} PUBLIC ngraph inference_engine_transformations
                                         PRIVATE mvnc openvino::itt)

    if(NOT STATIC_IE)
        target_link_libraries(${TARGET_NAME} PUBLIC inference_engine_legacy)
    endif()
endfunction()

add_common_target("vpu_common_lib" FALSE)

# Unit tests support for graph transformer
if(WIN32)
    add_common_target("vpu_common_lib_test_static" TRUE)
else()
    add_library("vpu_common_lib_test_static" ALIAS "vpu_common_lib")
endif()
