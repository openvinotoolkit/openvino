# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME compile_tool)
add_definitions(-DIN_OV_COMPONENT)

file(GLOB SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(${TARGET_NAME} ${SRCS})

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

ov_cpack_add_component(${OV_CPACK_COMP_CORE_TOOLS}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE})

if(CPACK_GENERATOR MATCHES "^(DEB|RPM)$")
    install(TARGETS compile_tool
            RUNTIME DESTINATION ${OV_CPACK_TOOLSDIR}
            COMPONENT ${OV_CPACK_COMP_CORE_TOOLS}
            ${OV_CPACK_COMP_CORE_TOOLS_EXCLUDE_ALL})
else()
    install(TARGETS compile_tool
            RUNTIME DESTINATION ${OV_CPACK_TOOLSDIR}/compile_tool
            COMPONENT ${OV_CPACK_COMP_CORE_TOOLS}
            ${OV_CPACK_COMP_CORE_TOOLS_EXCLUDE_ALL})
endif()
