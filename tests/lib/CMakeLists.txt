# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Search OpenVINO Runtime installed
find_package(PkgConfig QUIET)
# TODO: fix cross-compilation later
if(PkgConfig_FOUND AND NOT CMAKE_CROSSCOMPILING AND CMAKE_BUILD_TYPE STREQUAL "Release")
    pkg_search_module(openvino REQUIRED
                      IMPORTED_TARGET
                      openvino)
    set(ov_link_libraries PkgConfig::openvino)
else()
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    set(ov_link_libraries openvino::runtime)
endif()

add_subdirectory(src)
