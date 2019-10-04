# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# module to locate GNA libraries

if (WIN32)
    set(GNA_PLATFORM_DIR win64)
elseif (UNIX)
    set(GNA_PLATFORM_DIR linux)
else ()
    message(FATAL_ERROR "GNA not supported on this platform, only linux, and windows")
endif ()

set(libGNA_FOUND TRUE)

set(GNA_KERNEL_LIB_NAME gna)
set(GNA_LIBS_LIST
        "libGNA::API"
        "libGNA::KERNEL")

if (GNA_LIBRARY_VERSION STREQUAL "GNA1")
    # use old version of GNA Library from gna_20181120
    if (WIN32)
        set(GNA_LIB_DIR x64)
    else ()
        list(APPEND GNA_LIBS_LIST
                "libGNA::OLD_API_LIB")
        set(GNA_LIB_DIR lib)
        set(GNA_KERNEL_LIB_NAME gna_kernel)
    endif()
    set(libGNA_INCLUDE_DIRS "${GNA}/${GNA_PLATFORM_DIR}/include")
else()
    # use current version of GNA library
    set(GNA_LIB_DIR x64)
    set(libGNA_INCLUDE_DIRS "${GNA}/include")
endif()
set(libGNA_LIBRARIES_BASE_PATH ${GNA}/${GNA_PLATFORM_DIR}/${GNA_LIB_DIR})

add_library(libGNA::KERNEL SHARED IMPORTED)
find_library(GNA_KERNEL_LIBRARY
        ${GNA_KERNEL_LIB_NAME}
        HINTS
        ${libGNA_LIBRARIES_BASE_PATH})
set_target_properties(libGNA::KERNEL PROPERTIES IMPORTED_LOCATION ${GNA_KERNEL_LIBRARY})

if ((GNA_LIBRARY_VERSION STREQUAL "GNA1") AND (NOT WIN32))
    add_library(libGNA::OLD_API_LIB SHARED IMPORTED)
    find_library(GNA_API_LIBRARY
            gna_api
            HINTS
            ${libGNA_LIBRARIES_BASE_PATH})
    set_target_properties(libGNA::OLD_API_LIB PROPERTIES IMPORTED_LOCATION ${GNA_API_LIBRARY})
    target_link_libraries(libGNA::OLD_API_LIB INTERFACE libGNA::KERNEL)
    set_target_properties(libGNA::OLD_API_LIB PROPERTIES IMPORTED_NO_SONAME TRUE)
    set_target_properties(libGNA::KERNEL PROPERTIES IMPORTED_NO_SONAME TRUE)
endif()

add_library(libGNA::API INTERFACE IMPORTED)
set_property(TARGET libGNA::API PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${libGNA_INCLUDE_DIRS})

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
elseif(NOT GNA_LIBRARY_VERSION STREQUAL "GNA1")
    set_target_properties(libGNA PROPERTIES INTERFACE_LINK_OPTIONS "-Wl,-rpath-link,${libGNA_LIBRARIES_BASE_PATH}")
endif ()
