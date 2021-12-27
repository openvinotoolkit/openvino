# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# C++ tools
#

add_subdirectory(compile_tool)

#
# Python tools
#

# install deployment_manager

ie_cpack_add_component(deployment_manager REQUIRED)

install(DIRECTORY deployment_manager
        DESTINATION share/openvino/tools
        COMPONENT deployment_manager
        USE_SOURCE_PERMISSIONS)

# outbound requirements.txt files for openvino-dev package

set(REQUIREMENTS_IN "${CMAKE_CURRENT_SOURCE_DIR}/requirements_dev.in")
set(EXTRAS_LIST _ caffe kaldi mxnet onnx pytorch tensorflow tensorflow2)
ie_cpack_add_component(${OV_COMP_DEV_REQ_FILES})

foreach(EXTRAS ${EXTRAS_LIST})
    if(${EXTRAS} STREQUAL "_")
        set(REQUIREMENTS_OUT "requirements.txt")
        set(EXTRAS "")
    else()
        set(REQUIREMENTS_OUT "requirements_${EXTRAS}.txt")
        set(EXTRAS "[${EXTRAS}]")
    endif()
    configure_file(${REQUIREMENTS_IN} ${REQUIREMENTS_OUT})

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${REQUIREMENTS_OUT}
        DESTINATION share/openvino/tools
        COMPONENT ${OV_COMP_DEV_REQ_FILES}
        EXCLUDE_FROM_ALL)
endforeach()

add_subdirectory(mo)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/pot/version.txt ${CI_BUILD_NUMBER})

# build and install openvino-dev wheel

if(ENABLE_WHEEL)
    add_subdirectory(openvino_dev)
endif()
