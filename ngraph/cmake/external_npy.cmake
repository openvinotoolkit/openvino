# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install NPY ...
#------------------------------------------------------------------------------

SET(NPY_GIT_REPO_URL https://github.com/llohse/libnpy.git)

# Build for ninja

ExternalProject_Add(
        ext_libnpy
        PREFIX libnpy
        GIT_REPOSITORY ${NPY_GIT_REPO_URL}
        GIT_TAG "master"
        CONFIGURE_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        BUILD_COMMAND ""
)

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_libnpy SOURCE_DIR)
message("******${SOURCE_DIR}")

add_library(libnpy INTERFACE)
add_dependencies(libnpy ext_libnpy)
target_include_directories(libnpy SYSTEM INTERFACE
        ${SOURCE_DIR})
