########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(OpenCLHeaders_FIND_QUIETLY)
    set(OpenCLHeaders_MESSAGE_MODE VERBOSE)
else()
    set(OpenCLHeaders_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OpenCLHeadersTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${opencl-headers_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(OpenCLHeaders_VERSION_STRING "2023.04.17")
set(OpenCLHeaders_INCLUDE_DIRS ${opencl-headers_INCLUDE_DIRS_RELEASE} )
set(OpenCLHeaders_INCLUDE_DIR ${opencl-headers_INCLUDE_DIRS_RELEASE} )
set(OpenCLHeaders_LIBRARIES ${opencl-headers_LIBRARIES_RELEASE} )
set(OpenCLHeaders_DEFINITIONS ${opencl-headers_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${opencl-headers_BUILD_MODULES_PATHS_RELEASE} )
    message(${OpenCLHeaders_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


