# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_myriad_plugin")

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

if(CMAKE_COMPILER_IS_GNUCXX)
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
        mvnc inference_engine_legacy vpu_graph_transformer)

# MyriadPlugin is not safe to unload it at runtime
if((LINUX AND LINUX_OS_NAME MATCHES "Ubuntu") OR ARM)
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_OPTIONS "-Wl,-z,nodelete")
endif()

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# LTO

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install

if(LINUX)
    install(FILES ${mvnc_SOURCE_DIR}/src/97-myriad-usbboot.rules
            DESTINATION install_dependencies
            COMPONENT myriad)
endif()
