# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

########################################################################
#
#  Perform search of TBB package corresponding with specified search order.
#
#  TBBROOT var is set into external package path or has a default value
#  with IE own version of TBB. Search order is next:
#   1) ${TBBROOT}/cmake
#   2) ${TBBROOT} with IE own version of TBBConfig.cmake (actual for TBB < 2017.7)
#

# Path to IE own version of TBBConfig.cmake old TBB version without cmake config.
if(APPLE)
    set(IE_OWN_TBB_CONFIG tbb/mac)
elseif(UNIX)
    set(IE_OWN_TBB_CONFIG tbb/lnx)
elseif(WIN)
    set(IE_OWN_TBB_CONFIG tbb/win)
else()
    unset(IE_OWN_TBB_CONFIG)
endif()

find_package(TBB
    CONFIG
    PATHS ${TBBROOT}/cmake
          ${IEDevScripts_DIR}/${IE_OWN_TBB_CONFIG}
    NO_CMAKE_FIND_ROOT_PATH
    NO_DEFAULT_PATH
)

find_package_handle_standard_args(TBB CONFIG_MODE)
