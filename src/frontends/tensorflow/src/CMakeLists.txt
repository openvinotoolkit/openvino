# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME tensorflow
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to load and convert TensorFlow file format"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)

set(TARGET_NAME "${FRONTEND_NAME_PREFIX}tensorflow${FRONTEND_NAME_SUFFIX}")
target_link_libraries(${TARGET_NAME} PRIVATE openvino::frontend::tensorflow_common::dev)
target_include_directories(${TARGET_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/frontends/tensorflow_common/include>)

install(DIRECTORY ${OpenVINO_SOURCE_DIR}/src/frontends/tensorflow_common/include/openvino
        DESTINATION ${OV_CPACK_INCLUDEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        FILES_MATCHING PATTERN "*.hpp")