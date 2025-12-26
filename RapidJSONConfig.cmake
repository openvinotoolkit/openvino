########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(RapidJSON_FIND_QUIETLY)
    set(RapidJSON_MESSAGE_MODE VERBOSE)
else()
    set(RapidJSON_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/RapidJSONTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${rapidjson_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(RapidJSON_VERSION_STRING "cci.20220822")
set(RapidJSON_INCLUDE_DIRS ${rapidjson_INCLUDE_DIRS_RELEASE} )
set(RapidJSON_INCLUDE_DIR ${rapidjson_INCLUDE_DIRS_RELEASE} )
set(RapidJSON_LIBRARIES ${rapidjson_LIBRARIES_RELEASE} )
set(RapidJSON_DEFINITIONS ${rapidjson_DEFINITIONS_RELEASE} )


# Definition of extra CMake variables from cmake_extra_variables


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${rapidjson_BUILD_MODULES_PATHS_RELEASE} )
    message(${RapidJSON_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


