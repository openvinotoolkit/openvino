# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Install licensing
#

ie_cpack_add_component(${OV_CPACK_COMP_LICENSING} HIDDEN)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${OV_CPACK_DOCDIR}/licensing
        COMPONENT ${OV_CPACK_COMP_LICENSING}
        PATTERN CMakeLists.txt EXCLUDE
        PATTERN dev-third-party-programs.txt EXCLUDE
        PATTERN documentation-third-party-programs.txt EXCLUDE
        PATTERN third-party-programs.txt EXCLUDE)
