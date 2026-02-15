# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT TBBROOT)
    if(DEFINED ENV{TBBROOT})
        set (TBBROOT $ENV{TBBROOT})
    endif()
endif()

set(_tbb_root ${TBBROOT})

# detect version
find_file(_tbb_def_header tbb_stddef.h HINTS "${_tbb_root}/include/tbb")

if (_tbb_def_header)
    file(READ "${_tbb_def_header}" _tbb_def_content)
    string(REGEX MATCH "TBB_VERSION_MAJOR[ ]*[0-9]*" _tbb_version_major ${_tbb_def_content})
    string(REGEX MATCH "[0-9][0-9]*" _tbb_version_major ${_tbb_version_major})

    string(REGEX MATCH "TBB_VERSION_MINOR[ ]*[0-9]" _tbb_version_minor ${_tbb_def_content})
    string(REGEX MATCH "[0-9][0-9]*" _tbb_version_minor ${_tbb_version_minor})

    set(PACKAGE_VERSION_MAJOR ${_tbb_version_major})
    set(PACKAGE_VERSION_MINOR ${_tbb_version_minor})
    set(PACKAGE_VERSION_PATCH 0)
else()
    set(PACKAGE_VERSION_MAJOR 0)
    set(PACKAGE_VERSION_MINOR 0)
    set(PACKAGE_VERSION_PATCH 0)
endif()

set(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}")

set(PACKAGE_VERSION_EXACT False)
set(PACKAGE_VERSION_COMPATIBLE False)

if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT True)
    set(PACKAGE_VERSION_COMPATIBLE True)
endif()

if(PACKAGE_FIND_VERSION_MAJOR VERSION_LESS_EQUAL PACKAGE_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE True)
endif()
