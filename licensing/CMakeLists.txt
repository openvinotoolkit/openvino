# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Install licensing
#

ie_cpack_add_component(licensing)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION share/doc/openvino/licensing
        COMPONENT licensing
        PATTERN CMakeLists.txt EXCLUDE)
