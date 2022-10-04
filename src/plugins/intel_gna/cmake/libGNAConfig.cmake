# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# module to locate GNA libraries


set(libGNA_FOUND TRUE)

set(GNA_KERNEL_LIB_NAME gna CACHE STRING "" FORCE)

if (NOT libGNA_FIND_COMPONENTS)
    set(libGNA_FIND_COMPONENTS "API;KERNEL")
endif()

foreach (_gna_component ${libGNA_FIND_COMPONENTS})
    set(libGNA_${_gna_component}_FOUND FALSE)
    set(libGNA_FIND_REQUIRED_${_gna_component} TRUE)
endforeach()

set(libGNA_LIBRARIES_BASE_PATH ${GNA_PATH} CACHE STRING "" FORCE)

if(libGNA_FIND_REQUIRED_KERNEL)
    find_library(GNA_KERNEL_LIBRARY ${GNA_KERNEL_LIB_NAME}
                 HINTS ${libGNA_LIBRARIES_BASE_PATH}
                 NO_CMAKE_FIND_ROOT_PATH)

    if(GNA_KERNEL_LIBRARY)
        add_library(libGNA::KERNEL SHARED IMPORTED)
        set_target_properties(libGNA::KERNEL PROPERTIES IMPORTED_LOCATION ${GNA_KERNEL_LIBRARY})
        set(libGNA_KERNEL_FOUND TRUE)
    else()
        message(SEND_ERROR "GNA KERNEL library (${GNA_KERNEL_LIB_NAME}) was not found in ${libGNA_LIBRARIES_BASE_PATH}")
    endif()
endif()

if(libGNA_FIND_REQUIRED_API)
    find_path(libGNA_INCLUDE_DIRS gna2-api.h
              PATHS "${GNA_EXT_DIR}/include"
              NO_CMAKE_FIND_ROOT_PATH)
    if(libGNA_INCLUDE_DIRS)
        add_library(libGNA::API INTERFACE IMPORTED)
        set_target_properties(libGNA::API PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${libGNA_INCLUDE_DIRS})
        set(libGNA_API_FOUND TRUE)
    else()
        message(SEND_ERROR "GNA API headers (gna2-api.h) was not found in ${GNA_EXT_DIR}/include")
    endif()
endif()

add_library(libGNA INTERFACE IMPORTED)
foreach(_lib_name ${libGNA_FIND_COMPONENTS})
    set_property(TARGET libGNA APPEND PROPERTY INTERFACE_LINK_LIBRARIES libGNA::${_lib_name})
endforeach(_lib_name)

if (WIN32)
    if(libGNA_FIND_REQUIRED_KERNEL)
        set_target_properties(libGNA::KERNEL PROPERTIES
            IMPORTED_IMPLIB ${GNA_KERNEL_LIBRARY})
    endif()
else()
    set_target_properties(libGNA PROPERTIES INTERFACE_LINK_OPTIONS "-Wl,-rpath-link,${libGNA_LIBRARIES_BASE_PATH}")
endif ()
