# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h *.inc)

function(add_graph_transformer_target TARGET_NAME STATIC_IE)
    add_library(${TARGET_NAME} STATIC ${SOURCES})

    set_ie_threading_interface_for(${TARGET_NAME})

    # To avoid further TBB find_package action in next call of this function. Some version of TBB
    # has an issue with cmake config which lead to fail in case of multiple call of find_package
    # from one cmake script file.
    set(TBB_FOUND ${TBB_FOUND} PARENT_SCOPE)
    set(TBB_IMPORTED_TARGETS ${TBB_IMPORTED_TARGETS} PARENT_SCOPE)
    set(TBB_VERSION ${TBB_VERSION} PARENT_SCOPE)

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

    target_include_directories(${TARGET_NAME}
        SYSTEM PUBLIC
            $<TARGET_PROPERTY:pugixml,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
            "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include")

    if(STATIC_IE)
        target_link_libraries(${TARGET_NAME} PUBLIC vpu_common_lib_test_static)
    else()
        target_link_libraries(${TARGET_NAME} PUBLIC pugixml vpu_common_lib)
    endif()

    target_link_libraries(${TARGET_NAME} PUBLIC ${NGRAPH_LIBRARIES}
                                         PRIVATE openvino::itt)

    if(WIN32)
        target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)

        set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
    endif()

    if(NOT STATIC_IE)
        add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME} CUSTOM_FILTERS "+runtime/explicit")

        openvino_developer_export_targets(COMPONENT inference_engine_vpu TARGETS ${TARGET_NAME})
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endfunction()

add_graph_transformer_target("vpu_graph_transformer" FALSE)

# Unit tests support for graph transformer
if(WIN32)
    add_graph_transformer_target("vpu_graph_transformer_test_static" TRUE)
else()
    add_library("vpu_graph_transformer_test_static" ALIAS "vpu_graph_transformer")
endif()
