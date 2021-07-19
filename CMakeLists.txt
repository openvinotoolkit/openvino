# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(OpenVINO DESCRIPTION "OpenVINO toolkit")

set(IE_MAIN_SOURCE_DIR ${OpenVINO_SOURCE_DIR}/inference-engine)

find_package(IEDevScripts REQUIRED
             PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

include(CTest)
include(cmake/features.cmake)

# These options are shared with 3rdparty plugins by means of developer package
include(cmake/dependencies.cmake)

if(ENABLE_COVERAGE)
    include(cmake/coverage.cmake)
endif()

# resolving dependencies for the project
message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "OpenVINO_SOURCE_DIR ....     .......... " ${OpenVINO_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_BUILD_TYPE ...................... " ${CMAKE_BUILD_TYPE})

# remove file with exported developer targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/InferenceEngineTargets.cmake")
foreach(component IN LISTS openvino_export_components)
    file(REMOVE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
    unset(${component} CACHE)
endforeach()

#
# Build
#

function(openvino_developer_export_targets)
    cmake_parse_arguments(EXPORT "" "COMPONENT" "TARGETS" ${ARGN})

    if(EXPORT_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "openvino_developer_export_targets has unparsed arguments: ${EXPORT_UNPARSED_ARGUMENTS}")
    endif()

    set(${EXPORT_COMPONENT} "${${EXPORT_COMPONENT}};${EXPORT_TARGETS}")

    # to allow exporting of aliased targets with the original names
    foreach(target_name IN LISTS ${EXPORT_COMPONENT})
        if(TARGET "${target_name}")
            get_target_property(original_name ${target_name} ALIASED_TARGET)
            if(TARGET "${original_name}")
                message(STATUS "The name ${target_name} is an ALIAS for ${original_name}. "
                        "It will be exported to the InferenceEngineDeveloperPackage with the original name.")
                list(REMOVE_ITEM ${EXPORT_COMPONENT} ${target_name})
                list(APPEND ${EXPORT_COMPONENT} ${original_name})
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES ${EXPORT_COMPONENT})
    set(${EXPORT_COMPONENT} "${${EXPORT_COMPONENT}}" CACHE INTERNAL
        "A list of OpenVINO ${EXPORT_COMPONENT} exported targets" FORCE)

    list(APPEND openvino_export_components ${EXPORT_COMPONENT})
    list(REMOVE_DUPLICATES openvino_export_components)
    set(openvino_export_components "${openvino_export_components}" CACHE INTERNAL
        "A list of OpenVINO exported components" FORCE)
endfunction()

ie_cpack_add_component(ngraph REQUIRED)
ie_cpack_add_component(ngraph_dev REQUIRED DEPENDS ngraph)

add_subdirectory(thirdparty)
add_subdirectory(openvino)
add_subdirectory(ngraph)
add_subdirectory(inference-engine)

# for Template plugin
openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph_backend interpreter_backend)
include(cmake/extra_modules.cmake)

add_subdirectory(model-optimizer)
add_subdirectory(docs)
add_subdirectory(tools)
add_subdirectory(scripts)

#
# CPack
#

ie_cpack(${IE_CPACK_COMPONENTS_ALL})
