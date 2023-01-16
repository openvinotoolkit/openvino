# Copyright (C) 2018-2023 Intel Corporation
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

# for cross-compilation with gflags
find_package(Threads REQUIRED)

target_link_libraries(${TARGET_NAME} PRIVATE
    inference_engine
    inference_engine_plugin_api
    gflags
    ie_samples_utils
    Threads::Threads
)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME}
    FOLDER tools
)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# install

ie_cpack_add_component(${OV_CPACK_COMP_CORE_TOOLS}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE})

if(CPACK_GENERATOR MATCHES "^(DEB|RPM|CONDA-FORGE|BREW)$")
    install(TARGETS compile_tool
            RUNTIME DESTINATION ${OV_CPACK_TOOLSDIR}
            COMPONENT ${OV_CPACK_COMP_CORE_TOOLS})
    # TODO: convert this README.md to man page
else()
    install(TARGETS compile_tool
            RUNTIME DESTINATION ${OV_CPACK_TOOLSDIR}/compile_tool
            COMPONENT ${OV_CPACK_COMP_CORE_TOOLS})
    install(FILES README.md
            DESTINATION ${OV_CPACK_TOOLSDIR}/compile_tool
            COMPONENT ${OV_CPACK_COMP_CORE_TOOLS})
endif()
