# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

find_package(HddlUnite QUIET)
if(HddlUnite_FOUND)
    # test kmb_plugin directory
    if(EXISTS ${KMBPLUGIN_HOME}/include/)
        set(HDDL2_PARAMS_DIR ${KMBPLUGIN_HOME}/include/)
        set(HDDL2_DEFINITION USE_REMOTE_MEM)
        set(HDDL2_DEPENDENCE "HddlUnite::HddlUnite")
    else()
        message(WARNING "kmb_plugin/include could not find. Preallocating memory in remote device feature is disabled.")
    endif()
endif()

ie_add_sample(NAME benchmark_app
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader ${HDDL2_DEPENDENCE}
              OPENCV_DEPENDENCIES imgcodecs
              INCLUDE_DIRECTORIES ${HDDL2_PARAMS_DIR}
              MACRO_DEFINITIONS ${HDDL2_DEFINITION})
