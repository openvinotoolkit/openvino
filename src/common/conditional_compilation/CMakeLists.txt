# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME conditional_compilation)

add_library(${TARGET_NAME} INTERFACE)

add_library(openvino::conditional_compilation ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} INTERFACE openvino::itt)

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(SELECTIVE_BUILD STREQUAL "COLLECT")
    target_compile_definitions(${TARGET_NAME} INTERFACE SELECTIVE_BUILD_ANALYZER)
elseif(SELECTIVE_BUILD STREQUAL "ON")
    if(NOT DEFINED SELECTIVE_BUILD_STAT)
        message(FATAL_ERROR "In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected InelSEAPI statistics.\
 Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv")
    endif()
    find_package (PythonInterp 3 REQUIRED)

    file(GLOB STAT_FILES ${SELECTIVE_BUILD_STAT})

    target_compile_definitions(${TARGET_NAME} INTERFACE SELECTIVE_BUILD)

    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        # After disabling a block of code, some variables might be unused.
        target_compile_options(${TARGET_NAME} INTERFACE
                                -Wno-unused-function
                                -Wno-unused-parameter
                                -Wno-unused-local-typedefs)
    endif()

    set(GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/conditional_compilation_gen.h CACHE FILEPATH "")
    set(GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ccheader.py)

    add_custom_command(OUTPUT ${GENERATED_HEADER}
                       COMMAND ${PYTHON_EXECUTABLE} ${GENERATOR} --stat ${SELECTIVE_BUILD_STAT} --out ${GENERATED_HEADER}
                       DEPENDS ${STAT_FILES})
    add_custom_target(conditional_compilation_gen DEPENDS ${GENERATED_HEADER})
    add_dependencies(${TARGET_NAME} conditional_compilation_gen)

    ov_force_include(${TARGET_NAME} INTERFACE ${GENERATED_HEADER})
endif()

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

file(GLOB_RECURSE hdrs ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
add_clang_format_target(${TARGET_NAME}_clang FOR_SOURCES ${hdrs})
openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::conditional_compilation)
