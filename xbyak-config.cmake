########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(xbyak_FIND_QUIETLY)
    set(xbyak_MESSAGE_MODE VERBOSE)
else()
    set(xbyak_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/xbyakTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${xbyak_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(xbyak_VERSION_STRING "6.73")
set(xbyak_INCLUDE_DIRS ${xbyak_INCLUDE_DIRS_RELEASE} )
set(xbyak_INCLUDE_DIR ${xbyak_INCLUDE_DIRS_RELEASE} )
set(xbyak_LIBRARIES ${xbyak_LIBRARIES_RELEASE} )
set(xbyak_DEFINITIONS ${xbyak_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${xbyak_BUILD_MODULES_PATHS_RELEASE} )
    message(${xbyak_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


