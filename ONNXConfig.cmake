########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(ONNX_FIND_QUIETLY)
    set(ONNX_MESSAGE_MODE VERBOSE)
else()
    set(ONNX_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ONNXTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${onnx_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(ONNX_VERSION_STRING "1.17.0")
set(ONNX_INCLUDE_DIRS ${onnx_INCLUDE_DIRS_RELEASE} )
set(ONNX_INCLUDE_DIR ${onnx_INCLUDE_DIRS_RELEASE} )
set(ONNX_LIBRARIES ${onnx_LIBRARIES_RELEASE} )
set(ONNX_DEFINITIONS ${onnx_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${onnx_BUILD_MODULES_PATHS_RELEASE} )
    message(${ONNX_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


