########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(OpenCLHeadersCpp_FIND_QUIETLY)
    set(OpenCLHeadersCpp_MESSAGE_MODE VERBOSE)
else()
    set(OpenCLHeadersCpp_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OpenCLHeadersCppTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${opencl-clhpp-headers_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(OpenCLHeadersCpp_VERSION_STRING "2023.04.17")
set(OpenCLHeadersCpp_INCLUDE_DIRS ${opencl-clhpp-headers_INCLUDE_DIRS_RELEASE} )
set(OpenCLHeadersCpp_INCLUDE_DIR ${opencl-clhpp-headers_INCLUDE_DIRS_RELEASE} )
set(OpenCLHeadersCpp_LIBRARIES ${opencl-clhpp-headers_LIBRARIES_RELEASE} )
set(OpenCLHeadersCpp_DEFINITIONS ${opencl-clhpp-headers_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${opencl-clhpp-headers_BUILD_MODULES_PATHS_RELEASE} )
    message(${OpenCLHeadersCpp_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


