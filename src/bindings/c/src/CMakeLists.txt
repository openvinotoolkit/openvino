# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_c)

# Suppress warnings due to catch macro with legacy exception types
ov_disable_deprecated_warnings()

add_definitions(-DIN_OV_COMPONENT)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${OpenVINO_C_API_SOURCE_DIR}/include/openvino/*.h)

# create library
add_library(${TARGET_NAME} ${LEGACY_HEADERS} ${HEADERS} ${SOURCES})
add_library(openvino::runtime::c ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE openvino openvino::util)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OpenVINO_C_API_SOURCE_DIR}/include>)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${HEADERS} ${SOURCES})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_add_vs_version_file(NAME ${TARGET_NAME}
    FILEDESCRIPTION "OpenVINO C API Core Runtime library")

ov_set_install_rpath(${TARGET_NAME}
                     # openvino_c installed in the same directory as openvino
                     ${OV_CPACK_RUNTIMEDIR} ${OV_CPACK_RUNTIMEDIR})

# export

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME runtime::c)

ov_add_library_version(${TARGET_NAME})

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

# install

ov_cpack_add_component(${OV_CPACK_COMP_CORE_C} HIDDEN)
ov_cpack_add_component(${OV_CPACK_COMP_CORE_C_DEV} HIDDEN)

if(BUILD_SHARED_LIBS)
    set(archive_comp CORE_C_DEV)
else()
    set(archive_comp CORE_C)
endif()

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${OV_CPACK_COMP_CORE_C} ${OV_CPACK_COMP_CORE_C_EXCLUDE_ALL}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${OV_CPACK_COMP_${archive_comp}} ${OV_CPACK_COMP_${archive_comp}_EXCLUDE_ALL}
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${OV_CPACK_COMP_CORE_C} ${OV_CPACK_COMP_CORE_C_EXCLUDE_ALL}
        NAMELINK_COMPONENT ${OV_CPACK_COMP_LINKS} ${OV_CPACK_COMP_LINKS_EXCLUDE_ALL}
        INCLUDES DESTINATION ${OV_CPACK_INCLUDEDIR})

install(DIRECTORY ${OpenVINO_C_API_SOURCE_DIR}/include/openvino/
        DESTINATION ${OV_CPACK_INCLUDEDIR}/openvino
        COMPONENT ${OV_CPACK_COMP_CORE_C_DEV}
        ${OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL})
