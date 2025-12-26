########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(Snappy_FIND_QUIETLY)
    set(Snappy_MESSAGE_MODE VERBOSE)
else()
    set(Snappy_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/SnappyTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${snappy_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(Snappy_VERSION_STRING "1.1.10")
set(Snappy_INCLUDE_DIRS ${snappy_INCLUDE_DIRS_RELEASE} )
set(Snappy_INCLUDE_DIR ${snappy_INCLUDE_DIRS_RELEASE} )
set(Snappy_LIBRARIES ${snappy_LIBRARIES_RELEASE} )
set(Snappy_DEFINITIONS ${snappy_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${snappy_BUILD_MODULES_PATHS_RELEASE} )
    message(${Snappy_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


