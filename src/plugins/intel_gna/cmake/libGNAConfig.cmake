# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# The module defines several imported targets:
#
# - (Optional) libGNA::API
# - (Optional) libGNA::KERNEL
#
# And high-level imported interface target:
#
# - libGNA
#
# And the following variables:
#
# - libGNA_API_FOUND
# - libGNA_KERNEL_FOUND
#
# The example usage:
#
#  find_package(libGNA NO_MODULE COMPONENTS API KERNEL)
#

set(libGNA_FOUND ON)

set(GNA_KERNEL_LIB_NAME gna CACHE STRING "" FORCE)

if(NOT libGNA_FIND_COMPONENTS)
    set(libGNA_FIND_COMPONENTS "API;KERNEL")
endif()

foreach (_gna_component ${libGNA_FIND_COMPONENTS})
    set(libGNA_${_gna_component}_FOUND OFF)
    set(libGNA_FIND_REQUIRED_${_gna_component} ON)
endforeach()

set(libGNA_LIBRARIES_BASE_PATH ${GNA_PATH} CACHE STRING "" FORCE)

if(libGNA_FIND_REQUIRED_KERNEL AND NOT TARGET libGNA::KERNEL)
    find_library(GNA_KERNEL_LIBRARY ${GNA_KERNEL_LIB_NAME}
                 HINTS ${libGNA_LIBRARIES_BASE_PATH}
                 NO_CMAKE_FIND_ROOT_PATH)

    if(GNA_KERNEL_LIBRARY)
        add_library(libGNA::KERNEL SHARED IMPORTED)
        set_property(TARGET libGNA::KERNEL APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
        if(WIN32)
            set(gna_dll "${CMAKE_SHARED_LIBRARY_PREFIX}${GNA_KERNEL_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
            set_target_properties(libGNA::KERNEL PROPERTIES
                IMPORTED_LOCATION_RELEASE "${libGNA_LIBRARIES_BASE_PATH}/${gna_dll}"
                IMPORTED_IMPLIB_RELEASE "${GNA_KERNEL_LIBRARY}")
        else()
            set_target_properties(libGNA::KERNEL PROPERTIES
                IMPORTED_LOCATION_RELEASE "${GNA_KERNEL_LIBRARY}"
                INTERFACE_LINK_OPTIONS "-Wl,-rpath-link,${libGNA_LIBRARIES_BASE_PATH}")
        endif()
    else()
        message(SEND_ERROR "GNA KERNEL library (${GNA_KERNEL_LIB_NAME}) was not found in ${libGNA_LIBRARIES_BASE_PATH}")
    endif()
endif()

if(libGNA_FIND_REQUIRED_API AND NOT TARGET libGNA::API)
    find_path(libGNA_INCLUDE_DIRS gna2-api.h
              PATHS "${GNA_EXT_DIR}/include"
              NO_CMAKE_FIND_ROOT_PATH)
    if(libGNA_INCLUDE_DIRS)
        add_library(libGNA::API INTERFACE IMPORTED)
        set_target_properties(libGNA::API PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${libGNA_INCLUDE_DIRS}")
    else()
        message(SEND_ERROR "GNA API headers (gna2-api.h) was not found in ${GNA_EXT_DIR}/include")
    endif()
endif()

if(TARGET libGNA::KERNEL)
    set(libGNA_KERNEL_FOUND ON)
endif()

if(TARGET libGNA::API)
    set(libGNA_API_FOUND ON)
endif()

if(NOT TARGET libGNA)
    add_library(libGNA INTERFACE IMPORTED)
    foreach(_lib_name IN LISTS libGNA_FIND_COMPONENTS)
        set_property(TARGET libGNA APPEND PROPERTY INTERFACE_LINK_LIBRARIES libGNA::${_lib_name})
    endforeach()
endif()
