########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(protobuf_FIND_QUIETLY)
    set(protobuf_MESSAGE_MODE VERBOSE)
else()
    set(protobuf_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/protobufTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${protobuf_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(protobuf_VERSION_STRING "3.21.12")
set(protobuf_INCLUDE_DIRS ${protobuf_INCLUDE_DIRS_RELEASE} )
set(protobuf_INCLUDE_DIR ${protobuf_INCLUDE_DIRS_RELEASE} )
set(protobuf_LIBRARIES ${protobuf_LIBRARIES_RELEASE} )
set(protobuf_DEFINITIONS ${protobuf_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${protobuf_BUILD_MODULES_PATHS_RELEASE} )
    message(${protobuf_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


