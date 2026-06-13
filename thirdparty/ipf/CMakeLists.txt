# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# IPF (Intel Platform Framework) SDK wrapper
# Creates IPF::ClientApi IMPORTED target from the IPF SDK directory

if(NOT IPF_SDK_ROOT)
    message(FATAL_ERROR "IPF_SDK_ROOT is not set. Cannot configure IPF SDK.")
endif()

# Validate SDK structure
set(_ipf_include_dir "${IPF_SDK_ROOT}/include")
set(_ipf_include_thirdparty_dir "${IPF_SDK_ROOT}/include/thirdparty")
set(_ipf_lib_release "${IPF_SDK_ROOT}/lib/Release/ClientApiLib.lib")
set(_ipf_lib_debug "${IPF_SDK_ROOT}/lib/Debug/ClientApiLib.lib")

if(NOT EXISTS "${_ipf_include_dir}/ClientApi.h")
    message(FATAL_ERROR "IPF SDK header not found: ${_ipf_include_dir}/ClientApi.h\n"
                        "Please verify IPF_SDK_ROOT points to a valid IPF SDK directory.")
endif()

# Create IMPORTED target
add_library(IPF::ClientApi STATIC IMPORTED GLOBAL)

set_target_properties(IPF::ClientApi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_ipf_include_dir};${_ipf_include_thirdparty_dir}")

if(EXISTS "${_ipf_lib_release}")
    set_property(TARGET IPF::ClientApi PROPERTY IMPORTED_LOCATION_RELEASE "${_ipf_lib_release}")
endif()
if(EXISTS "${_ipf_lib_debug}")
    set_property(TARGET IPF::ClientApi PROPERTY IMPORTED_LOCATION_DEBUG "${_ipf_lib_debug}")
endif()

# Set default config mapping so single-config generators work
if(EXISTS "${_ipf_lib_release}")
    set_property(TARGET IPF::ClientApi PROPERTY IMPORTED_LOCATION "${_ipf_lib_release}")
    set_property(TARGET IPF::ClientApi PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set_property(TARGET IPF::ClientApi PROPERTY MAP_IMPORTED_CONFIG_MINSIZEREL Release)
endif()

unset(_ipf_include_dir)
unset(_ipf_include_thirdparty_dir)
unset(_ipf_lib_release)
unset(_ipf_lib_debug)

message(STATUS "IPF SDK found at: ${IPF_SDK_ROOT}")
