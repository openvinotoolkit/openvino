########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(OpenCLICDLoader_FIND_QUIETLY)
    set(OpenCLICDLoader_MESSAGE_MODE VERBOSE)
else()
    set(OpenCLICDLoader_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OpenCLICDLoaderTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${opencl-icd-loader_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(OpenCLICDLoader_VERSION_STRING "2023.04.17")
set(OpenCLICDLoader_INCLUDE_DIRS ${opencl-icd-loader_INCLUDE_DIRS_RELEASE} )
set(OpenCLICDLoader_INCLUDE_DIR ${opencl-icd-loader_INCLUDE_DIRS_RELEASE} )
set(OpenCLICDLoader_LIBRARIES ${opencl-icd-loader_LIBRARIES_RELEASE} )
set(OpenCLICDLoader_DEFINITIONS ${opencl-icd-loader_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${opencl-icd-loader_BUILD_MODULES_PATHS_RELEASE} )
    message(${OpenCLICDLoader_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


