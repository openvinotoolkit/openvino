# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_common")

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

# Add include path to so_extension.hpp
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend_manager.cpp
                    APPEND PROPERTY INCLUDE_DIRECTORIES "${OpenVINO_SOURCE_DIR}/src/core/src/")

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# create library

add_library(${TARGET_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(ngraph::${TARGET_NAME} ALIAS ${TARGET_NAME})
add_library(openvino::frontend::common ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
    PRIVATE
        # for ov_frontends.hpp in static build
        "${CMAKE_CURRENT_BINARY_DIR}/src")

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS} openvino::util PUBLIC ngraph)
set_property(SOURCE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin_loader.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend_manager.cpp
        APPEND PROPERTY COMPILE_DEFINITIONS
        FRONTEND_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
        FRONTEND_LIB_SUFFIX="${FRONTEND_NAME_SUFFIX}${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    INCLUDE_DIRECTORY "${FRONTEND_INCLUDE_DIR}"
                    ADDITIONAL_INCLUDE_DIRECTORIES
                        $<TARGET_PROPERTY:ngraph,INTERFACE_INCLUDE_DIRECTORIES>)

ie_add_vs_version_file(NAME ${TARGET_NAME}
                       FILEDESCRIPTION "Manager of OpenVINO Frontends")
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

# create static library if needed

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME}_static STATIC ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
    add_library(${TARGET_NAME}::static ALIAS ${TARGET_NAME}_static)
    target_link_libraries(${TARGET_NAME}_static PRIVATE ${CMAKE_DL_LIBS} openvino::util PUBLIC ngraph)
    target_include_directories(${TARGET_NAME}_static PUBLIC
        $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>)
    target_include_directories(${TARGET_NAME}_static PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${TARGET_NAME}_static PUBLIC USE_STATIC_FRONTEND_COMMON)
    ov_install_static_lib(${TARGET_NAME}_static core)
else()
    add_library(${TARGET_NAME}::static ALIAS ${TARGET_NAME})
endif()

openvino_developer_export_targets(COMPONENT core TARGETS ${TARGET_NAME})

# Installation rules for shared version only

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::common)

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)

install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/common
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT core_dev
        FILES_MATCHING PATTERN "*.hpp")

install(DIRECTORY ${FRONTEND_INCLUDE_DIR}/
        DESTINATION ${FRONTEND_INSTALL_INCLUDE}
        COMPONENT core_dev
        FILES_MATCHING PATTERN "manager.hpp")

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
