########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(pybind11_FIND_QUIETLY)
    set(pybind11_MESSAGE_MODE VERBOSE)
else()
    set(pybind11_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/pybind11Targets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${pybind11_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(pybind11_VERSION_STRING "3.0.1")
set(pybind11_INCLUDE_DIRS ${pybind11_INCLUDE_DIRS_RELEASE} )
set(pybind11_INCLUDE_DIR ${pybind11_INCLUDE_DIRS_RELEASE} )
set(pybind11_LIBRARIES ${pybind11_LIBRARIES_RELEASE} )
set(pybind11_DEFINITIONS ${pybind11_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${pybind11_BUILD_MODULES_PATHS_RELEASE} )
    message(${pybind11_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


