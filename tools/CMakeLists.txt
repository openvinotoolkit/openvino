# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# C++ tools
#

add_subdirectory(compile_tool)

if(ENABLE_SAMPLES)
    add_subdirectory(legacy/benchmark_app)
endif()

#
# Python tools
#

# install deployment_manager

ie_cpack_add_component(deployment_manager REQUIRED)

install(DIRECTORY deployment_manager
        DESTINATION tools
        COMPONENT deployment_manager
        USE_SOURCE_PERMISSIONS)

# outbound requirements.txt files for openvino-dev package
if(DEFINED ENV{CI_BUILD_DEV_TAG} AND NOT $ENV{CI_BUILD_DEV_TAG} STREQUAL "")
    set(WHEEL_VERSION "${OpenVINO_VERSION}.$ENV{CI_BUILD_DEV_TAG}")
else()
    set(WHEEL_VERSION ${OpenVINO_VERSION})
endif()

set(REQUIREMENTS_IN "${CMAKE_CURRENT_SOURCE_DIR}/requirements_dev.in")
set(EXTRAS_LIST _ caffe kaldi mxnet onnx pytorch tensorflow tensorflow2)
ie_cpack_add_component(openvino_dev_req_files)

foreach(EXTRAS IN LISTS EXTRAS_LIST)
    if(EXTRAS STREQUAL "_")
        set(REQUIREMENTS_OUT "requirements.txt")
        set(EXTRAS "")
    else()
        set(REQUIREMENTS_OUT "requirements_${EXTRAS}.txt")
        set(EXTRAS "[${EXTRAS}]")
    endif()
    configure_file(${REQUIREMENTS_IN} ${REQUIREMENTS_OUT})

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${REQUIREMENTS_OUT}
            DESTINATION tools
            COMPONENT openvino_dev_req_files
            EXCLUDE_FROM_ALL)
endforeach()

add_subdirectory(mo)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pot/openvino/tools/pot/version.txt.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/pot/openvino/tools/pot/version.txt" @ONLY)

# build and install openvino-dev wheel

if(ENABLE_WHEEL)
    add_subdirectory(openvino_dev)
endif()
