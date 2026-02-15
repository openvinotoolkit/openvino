# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

########################################################################
#
#  Perform search of TBB package corresponding with specified search order.
#
#  TBBROOT var is set into external package path or has a default value
#  with OV own version of TBB. Search order is next:
#   1) ${TBBROOT}/cmake
#   2) ${TBBROOT} with OV own version of TBBConfig.cmake (actual for TBB < 2017.7)
#

# Path to OV own version of TBBConfig.cmake old TBB version without cmake config.
if(APPLE)
    set(_OV_OWN_TBB_CONFIG tbb/mac)
elseif(UNIX)
    set(_OV_OWN_TBB_CONFIG tbb/lnx)
elseif(WIN)
    set(_OV_OWN_TBB_CONFIG tbb/win)
else()
    unset(_OV_OWN_TBB_CONFIG)
endif()

unset(TBB_DIR)
unset(TBB_DIR CACHE)

find_package(TBB
    2017.0
    CONFIG
    PATHS "${OpenVINODeveloperScripts_DIR}/${_OV_OWN_TBB_CONFIG}"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(TBB CONFIG_MODE)

unset(_OV_OWN_TBB_CONFIG)
