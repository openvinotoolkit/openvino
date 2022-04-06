# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Install licensing
#

ie_cpack_add_component(licensing)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION docs/licensing
        COMPONENT licensing
        PATTERN CMakeLists.txt EXCLUDE
        PATTERN dev-third-party-programs.txt EXCLUDE
        PATTERN documentation-third-party-programs.txt EXCLUDE
        PATTERN third-party-programs.txt EXCLUDE)
