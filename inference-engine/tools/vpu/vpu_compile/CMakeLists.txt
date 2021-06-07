# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME myriad_compile)

file(GLOB SRCS
    ${IE_MAIN_SOURCE_DIR}/tools/vpu/common/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(${TARGET_NAME} ${SRCS})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME} PRIVATE
        "-Wall"
    )
endif()

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    ${IE_MAIN_SOURCE_DIR}/tools/vpu/common
    ${IE_MAIN_SOURCE_DIR}/samples/common
)

target_link_libraries(${TARGET_NAME} PRIVATE
    inference_engine vpu_graph_transformer
    gflags
    ie_samples_utils
)

add_dependencies(${TARGET_NAME} myriadPlugin)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME}
    FOLDER tools
)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# install

ie_cpack_add_component(myriad_dev DEPENDS myriad)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH}
        COMPONENT myriad_dev)
