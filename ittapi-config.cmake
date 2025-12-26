########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(ittapi_FIND_QUIETLY)
    set(ittapi_MESSAGE_MODE VERBOSE)
else()
    set(ittapi_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ittapiTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${ittapi_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(ittapi_VERSION_STRING "3.24.0")
set(ittapi_INCLUDE_DIRS ${ittapi_INCLUDE_DIRS_RELEASE} )
set(ittapi_INCLUDE_DIR ${ittapi_INCLUDE_DIRS_RELEASE} )
set(ittapi_LIBRARIES ${ittapi_LIBRARIES_RELEASE} )
set(ittapi_DEFINITIONS ${ittapi_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${ittapi_BUILD_MODULES_PATHS_RELEASE} )
    message(${ittapi_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


