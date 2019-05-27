# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "mvnc")

if(NOT WIN32)
    find_package(Threads REQUIRED)

    find_path(LIBUSB_INCLUDE_DIR NAMES libusb.h PATH_SUFFIXES "include" "libusb" "libusb-1.0")
    find_library(LIBUSB_LIBRARY NAMES usb-1.0 PATH_SUFFIXES "lib")

    if(NOT LIBUSB_INCLUDE_DIR OR NOT LIBUSB_LIBRARY)
        message(FATAL_ERROR "libusb it required for Myriad plugin build")
    endif()
endif()

file(GLOB MVNC_SOURCES "include/*" "src/*")

# Watchdog would  be disabled on PCIE configuration
if (NOT ENABLE_MYX_PCIE)
    file(GLOB WATCHDOG_SOURCES "../watchdog/*")
endif()

# FIXME: WIN_PTHREAD also should be built as a library
if(WIN32)
    file(GLOB USB_WIN_SOURCES "../USB_WIN/*")
    file(GLOB WIN_PTHREAD_SOURCES "../WinPthread/*")
    list(APPEND ${MVNC_SOURCES} ${USB_WIN_SOURCES} ${WIN_PTHREAD_SOURCES})
endif()

add_library(${TARGET_NAME} STATIC ${MVNC_SOURCES} ${WATCHDOG_SOURCES})

target_include_directories(${TARGET_NAME}
    PUBLIC
        "include"
    PRIVATE
        "../watchdog")

if(WIN32)
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "../USB_WIN"
            "../WinPthread")
endif()

if(UNIX)
    target_include_directories(${TARGET_NAME}
        PRIVATE
            "${LIBUSB_INCLUDE_DIR}")
endif()

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        __PC__
        HAVE_STRUCT_TIMESPEC
        _CRT_SECURE_NO_WARNINGS)

if (ENABLE_MYX_PCIE)
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_PCIE)
else ()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_USB_VSC)
endif()

if (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
endif()

if(NOT WIN32)
    target_compile_options(${TARGET_NAME}
        PRIVATE
            -MMD
            -MP
            -Wformat
            -Wformat-security
            -Wall)
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.8)
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

