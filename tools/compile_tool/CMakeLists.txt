# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME compile_tool)

file(GLOB SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(${TARGET_NAME} ${SRCS})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    ${OpenVINO_SOURCE_DIR}/src/plugins/intel_myriad/graph_transformer/include
    ${OpenVINO_SOURCE_DIR}/src/plugins/intel_myriad/common/include
)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    inference_engine
    inference_engine_plugin_api
    gflags
    ie_samples_utils
)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME}
    FOLDER tools
)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# install

ie_cpack_add_component(core_tools DEPENDS core)

install(TARGETS compile_tool
        RUNTIME DESTINATION tools/compile_tool
        COMPONENT core_tools)

install(FILES README.md
        DESTINATION tools/compile_tool
        COMPONENT core_tools)
