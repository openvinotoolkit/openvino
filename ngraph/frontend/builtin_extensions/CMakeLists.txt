# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_builtin_extensions")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

# Create library
add_library(${TARGET_NAME} SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS})
add_library(openvino::frontend::frontend_builtin_extensions ALIAS ${TARGET_NAME})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
            FILEDESCRIPTION "Set of separatelly compiled and conditionally loaded extensions for FrontEnds")
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ngraph::builder inference_engine_transformations frontend_manager)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::frontend_builtin_extensions)
install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ngraph
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ngraph
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ngraph)

install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/builtin_extensions
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT ngraph_dev
        FILES_MATCHING PATTERN "*.hpp")

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
        APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
