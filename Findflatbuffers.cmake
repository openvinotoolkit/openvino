########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(flatbuffers_FIND_QUIETLY)
    set(flatbuffers_MESSAGE_MODE VERBOSE)
else()
    set(flatbuffers_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/module-flatbuffersTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${flatbuffers_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(flatbuffers_VERSION_STRING "23.5.26")
set(flatbuffers_INCLUDE_DIRS ${flatbuffers_INCLUDE_DIRS_RELEASE} )
set(flatbuffers_INCLUDE_DIR ${flatbuffers_INCLUDE_DIRS_RELEASE} )
set(flatbuffers_LIBRARIES ${flatbuffers_LIBRARIES_RELEASE} )
set(flatbuffers_DEFINITIONS ${flatbuffers_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${flatbuffers_BUILD_MODULES_PATHS_RELEASE} )
    message(${flatbuffers_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


include(FindPackageHandleStandardArgs)
set(flatbuffers_FOUND 1)
set(flatbuffers_VERSION "23.5.26")

find_package_handle_standard_args(flatbuffers
                                  REQUIRED_VARS flatbuffers_VERSION
                                  VERSION_VAR flatbuffers_VERSION)
mark_as_advanced(flatbuffers_FOUND flatbuffers_VERSION)

set(flatbuffers_FOUND 1)
set(flatbuffers_VERSION "23.5.26")
mark_as_advanced(flatbuffers_FOUND flatbuffers_VERSION)

