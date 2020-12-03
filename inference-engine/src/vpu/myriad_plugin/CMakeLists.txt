# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "myriadPlugin")

file(GLOB_RECURSE SOURCES *.cpp *.hpp *.h)

ie_add_plugin(
    NAME ${TARGET_NAME}
    DEVICE_NAME "MYRIAD"
    SOURCES ${SOURCES}
    VERSION_DEFINES_FOR api/myriad_api.cpp
    CPPLINT_FILTERS "+runtime/explicit"
)

add_dependencies(${TARGET_NAME} vpu_copy_firmware)
if(TARGET vpu_compile_custom_kernels)
    add_dependencies(${TARGET_NAME} vpu_compile_custom_kernels)
endif()

target_include_directories(${TARGET_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -Wall)
endif()

if (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
endif()

# NOTE:
# sem_* POSIX functions must be overriden on OSX,
# these functions come from "mvnc" target,
# "mvnc" must be the first library in the link list
target_link_libraries(${TARGET_NAME}
    PRIVATE
        mvnc inference_engine inference_engine_legacy vpu_graph_transformer)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

ie_add_manifest(TARGET_NAME ${TARGET_NAME}
                DEPENDENCIES ${NGRAPH_LIBRARIES} inference_engine_legacy
                             inference_engine_transformations)

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install
if (LINUX)
    install(FILES ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/src/97-myriad-usbboot.rules
        DESTINATION ${IE_CPACK_IE_DIR}/external
        COMPONENT myriad)
endif()
