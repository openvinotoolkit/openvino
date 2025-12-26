########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(OpenCL_FIND_QUIETLY)
    set(OpenCL_MESSAGE_MODE VERBOSE)
else()
    set(OpenCL_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-OpenCLTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${opencl-icd-loader_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(OpenCL_VERSION_STRING "2023.04.17")
set(OpenCL_INCLUDE_DIRS ${opencl-icd-loader_INCLUDE_DIRS_RELEASE} )
set(OpenCL_INCLUDE_DIR ${opencl-icd-loader_INCLUDE_DIRS_RELEASE} )
set(OpenCL_LIBRARIES ${opencl-icd-loader_LIBRARIES_RELEASE} )
set(OpenCL_DEFINITIONS ${opencl-icd-loader_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${opencl-icd-loader_BUILD_MODULES_PATHS_RELEASE} )
    message(${OpenCL_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


include(FindPackageHandleStandardArgs)
set(OpenCL_FOUND 1)
set(OpenCL_VERSION "2023.04.17")

find_package_handle_standard_args(OpenCL
                                  REQUIRED_VARS OpenCL_VERSION
                                  VERSION_VAR OpenCL_VERSION)
mark_as_advanced(OpenCL_FOUND OpenCL_VERSION)

set(OpenCL_FOUND 1)
set(OpenCL_VERSION "2023.04.17")
mark_as_advanced(OpenCL_FOUND OpenCL_VERSION)

