# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(ittapi)
add_subdirectory(itt_collector)
add_subdirectory(xbyak EXCLUDE_FROM_ALL)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)

#
# Install
#

ie_cpack_add_component(cpp_samples_deps)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gflags
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zlib
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

#
# Export
#

openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags xbyak)
