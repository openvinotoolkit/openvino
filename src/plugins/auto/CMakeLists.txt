# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_AUTO AND NOT ENABLE_MULTI)
    return()
endif()

set (TARGET_NAME "openvino_auto_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp)

# WA for issue 141313
if(LINUX AND CMAKE_COMPILE_WARNING_AS_ERROR AND ENABLE_SYSTEM_TBB)
    execute_process(COMMAND lsb_release -a OUTPUT_VARIABLE output ERROR_QUIET)
    string(REGEX MATCH "Distributor ID:[ \t]+([^\n]+)" match_name ${output})

    if(match_name)
        set(PRETTY_NAME "${CMAKE_MATCH_1}")
    endif()

    string(REGEX MATCH "Release:[ \t]+([0-9.]+)" match_version ${output})

    if(match_version)
        set(PRETTY_VERSION "${CMAKE_MATCH_1}")
    endif()

    if(PRETTY_NAME MATCHES "Ubuntu" AND PRETTY_VERSION MATCHES "24.*")
        ov_add_compiler_flags(-Wno-stringop-overflow)
    endif()
endif()

if(ENABLE_AUTO AND ENABLE_MULTI)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)

    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  PSEUDO_PLUGIN_FOR "MULTI")
elseif(ENABLE_AUTO)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)
elseif(ENABLE_MULTI)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)
endif()

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
