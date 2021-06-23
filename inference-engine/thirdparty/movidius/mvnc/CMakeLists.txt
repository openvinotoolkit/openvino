# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "mvnc")

include(${XLINK_DIR}/XLink.cmake)

file(GLOB_RECURSE MVNC_SOURCES "include/*" "src/*")

set(WATCHDOG_INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/watchdog/)

add_library(${TARGET_NAME} STATIC ${MVNC_SOURCES})

target_include_directories(${TARGET_NAME}
        PUBLIC
        "include"
        ${WATCHDOG_INCLUDE}
        PRIVATE
        ${XLINK_INCLUDE}
        ${XLINK_PLATFORM_INCLUDE})

target_compile_definitions(${TARGET_NAME}
        PRIVATE
        __PC__
        HAVE_STRUCT_TIMESPEC
        _CRT_SECURE_NO_WARNINGS
        USE_USB_VSC)

if (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
endif()

if(NOT WIN32)
    target_compile_options(${TARGET_NAME}
            PRIVATE
            -Wformat
            -Wformat-security
            -Wall)
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.8.5)
        target_compile_options(${TARGET_NAME}
                PRIVATE
                -fstack-protector-strong)
    else()
        target_compile_options(${TARGET_NAME}
                PRIVATE
                -fstack-protector)
    endif()

    set_property(TARGET ${TARGET_NAME}
            PROPERTY LINK_FLAGS
            -z noexecstack
            -z relro
            -z now)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE
        COMPILE_PDB_NAME ${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
        PRIVATE
        XLink)

if(NOT WIN32)
    target_link_libraries(${TARGET_NAME}
            PUBLIC
            Threads::Threads
            ${CMAKE_DL_LIBS}
            ${LIBUSB_LIBRARY})
endif()

# TODO: remove once all options are migrated
openvino_developer_export_targets(COMPONENT inference_engine_vpu TARGETS ${TARGET_NAME})

if(ENABLE_TESTS AND ENABLE_MYRIAD_MVNC_TESTS)
    add_subdirectory(tests)
endif()
