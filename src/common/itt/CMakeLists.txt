# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_itt)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.hpp" "include/*.hpp")

add_library(${TARGET_NAME} STATIC ${SOURCES})

add_library(openvino::itt ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME itt)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::util)

if(ENABLE_PROFILING_ITT)
    if(TARGET ittapi::ittnotify)
        set(itt_dependency ittapi::ittnotify)
    else()
        set(itt_dependency ittapi::ittapi)
    endif()
    target_link_libraries(${TARGET_NAME} PUBLIC ${itt_dependency})

    if(ENABLE_PROFILING_FILTER STREQUAL "ALL")
        target_compile_definitions(${TARGET_NAME} PUBLIC
            ENABLE_PROFILING_ALL
            ENABLE_PROFILING_FIRST_INFERENCE)
    elseif(ENABLE_PROFILING_FILTER STREQUAL "FIRST_INFERENCE")
        target_compile_definitions(${TARGET_NAME} PUBLIC
            ENABLE_PROFILING_FIRST_INFERENCE)
    else()
        message(FATAL_ERROR "The ${ENABLE_PROFILING_FILTER} profiling filter isn't supported")
    endif()
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET openvino::itt
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")
