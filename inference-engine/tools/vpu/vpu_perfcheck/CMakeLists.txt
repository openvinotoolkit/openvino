# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(add_perfcheck_target TARGET_NAME PLUGIN_NAME)
    file(GLOB SOURCES *.cpp)

    add_executable(${TARGET_NAME} ${SOURCES})

    # TODO: enable some day and fix all warnings
#    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#        target_compile_options(${TARGET_NAME}
#            PRIVATE
#                "-Wall")
#    endif()

    target_include_directories(${TARGET_NAME}
        SYSTEM PRIVATE
            "${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include"
            "${IE_MAIN_SOURCE_DIR}/src/plugin_api"
            "${IE_MAIN_SOURCE_DIR}/samples/common/format_reader")

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            inference_engine format_reader
            Threads::Threads
            ie_samples_utils)

    add_dependencies(${TARGET_NAME}
        ${PLUGIN_NAME} ${ARGN})

    set_target_properties(${TARGET_NAME} PROPERTIES
        COMPILE_PDB_NAME ${TARGET_NAME}
        FOLDER tools)

    add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endfunction()

if(ENABLE_MYRIAD)
    add_perfcheck_target(myriad_perfcheck myriadPlugin)

    ie_cpack_add_component(myriad_tools DEPENDS myriad)

    install(TARGETS myriad_perfcheck
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH}
            COMPONENT myriad_tools)
endif()
