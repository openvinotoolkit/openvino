########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(Protobuf_FIND_QUIETLY)
    set(Protobuf_MESSAGE_MODE VERBOSE)
else()
    set(Protobuf_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-ProtobufTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${protobuf_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(Protobuf_VERSION_STRING "3.21.12")
set(Protobuf_INCLUDE_DIRS ${protobuf_INCLUDE_DIRS_RELEASE} )
set(Protobuf_INCLUDE_DIR ${protobuf_INCLUDE_DIRS_RELEASE} )
set(Protobuf_LIBRARIES ${protobuf_LIBRARIES_RELEASE} )
set(Protobuf_DEFINITIONS ${protobuf_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${protobuf_BUILD_MODULES_PATHS_RELEASE} )
    message(${Protobuf_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


include(FindPackageHandleStandardArgs)
set(Protobuf_FOUND 1)
set(Protobuf_VERSION "3.21.12")

find_package_handle_standard_args(Protobuf
                                  REQUIRED_VARS Protobuf_VERSION
                                  VERSION_VAR Protobuf_VERSION)
mark_as_advanced(Protobuf_FOUND Protobuf_VERSION)

set(Protobuf_FOUND 1)
set(Protobuf_VERSION "3.21.12")
mark_as_advanced(Protobuf_FOUND Protobuf_VERSION)

