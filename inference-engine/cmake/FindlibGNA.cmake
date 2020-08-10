# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# module to locate GNA libraries

if (WIN32)
    set(GNA_PLATFORM_DIR win64 CACHE STRING "" FORCE)
elseif (UNIX)
    set(GNA_PLATFORM_DIR linux CACHE STRING "" FORCE)
else ()
    message(FATAL_ERROR "GNA not supported on this platform, only linux, and windows")
endif ()

set(libGNA_FOUND TRUE)

set(GNA_LIBS_LIST
        "libGNA::API"
        "libGNA::KERNEL")
set(libGNA_LIBRARIES_BASE_PATH "${GNA}/${GNA_PLATFORM_DIR}/x64" CACHE STRING "" FORCE)

add_library(libGNA::KERNEL SHARED IMPORTED)
find_library(GNA_KERNEL_LIBRARY
        gna
        HINTS
        ${libGNA_LIBRARIES_BASE_PATH})
set_target_properties(libGNA::KERNEL PROPERTIES IMPORTED_LOCATION ${GNA_KERNEL_LIBRARY})

add_library(libGNA::API INTERFACE IMPORTED)
set_property(TARGET libGNA::API PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${GNA}/include")

add_library(libGNA INTERFACE IMPORTED)
foreach(_lib_name ${GNA_LIBS_LIST})
    set_property(TARGET libGNA APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${_lib_name})

    get_target_property(_target_type ${_lib_name} TYPE)
    if (${_target_type} STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_target_location ${_lib_name} INTERFACE_INCLUDE_DIRECTORIES)
    else()
        get_target_property(_target_location ${_lib_name} IMPORTED_LOCATION)
    endif ()
    message(STATUS "${_lib_name} ${_target_type} : ${_target_location}")
endforeach(_lib_name)

if (WIN32)
    set_target_properties(libGNA::KERNEL PROPERTIES
        IMPORTED_IMPLIB ${GNA_KERNEL_LIBRARY})
else()
    set_target_properties(libGNA PROPERTIES INTERFACE_LINK_OPTIONS "-Wl,-rpath-link,${libGNA_LIBRARIES_BASE_PATH}")
endif ()
