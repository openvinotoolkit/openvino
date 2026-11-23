# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:plugin]
set(TARGET_NAME "openvino_template_plugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ov_option(ENABLE_TEMPLATE_UTILS "Build template plugin utils library" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DTEMPLATE_UTILS_ENABLED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

if (NOT ENABLE_TEMPLATE_REGISTRATION)
    # Skip install and registration of template component
    set(skip_plugin SKIP_INSTALL SKIP_REGISTRATION)
endif()

# adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "TEMPLATE"
              SOURCES ${SOURCES} ${HEADERS}
              ${skip_plugin}
              VERSION_DEFINES_FOR plugin.cpp
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ov_mark_target_as_cc(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${OpenVINOTemplatePlugin_SOURCE_DIR}/include")

# link common OpenVINO Runtime libraries
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::interpreter_backend
    openvino::reference)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (ENABLE_TEMPLATE_REGISTRATION)
    # Update the plugins.xml file
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
endif()# [cmake:plugin]

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

if(ENABLE_TEMPLATE_UTILS)
    add_library(template_utils SHARED ${CMAKE_CURRENT_SOURCE_DIR}/utils/template_utils.cpp)
    target_link_libraries(template_utils PUBLIC openvino::runtime openvino::pugixml)
    target_link_libraries(openvino PUBLIC template_utils)
    install(TARGETS template_utils LIBRARY DESTINATION lib)
endif()
