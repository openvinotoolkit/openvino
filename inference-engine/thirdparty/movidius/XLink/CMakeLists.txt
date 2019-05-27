# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "XLink")

if(NOT WIN32)
    find_package(Threads REQUIRED)

    find_path(LIBUSB_INCLUDE_DIR NAMES libusb.h PATH_SUFFIXES "include" "libusb" "libusb-1.0")
    find_library(LIBUSB_LIBRARY NAMES usb-1.0 PATH_SUFFIXES "lib")

    if(NOT LIBUSB_INCLUDE_DIR OR NOT LIBUSB_LIBRARY)
        message(FATAL_ERROR "libusb is required")
    endif()
endif()

file(GLOB_RECURSE SOURCES *.c *.h)

# FIXME: WIN_PTHREAD also should be built as a library
if(WIN32)
    file(GLOB USB_WIN_SOURCES "../USB_WIN/*")
    file(GLOB WIN_PTHREAD_SOURCES "../WinPthread/*")
    list(APPEND SOURCES ${USB_WIN_SOURCES} ${WIN_PTHREAD_SOURCES})
endif()

add_library(${TARGET_NAME} STATIC ${SOURCES})

if(WIN32)
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "../USB_WIN"
            "../WinPthread")
else()
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "${LIBUSB_INCLUDE_DIR}")
endif()

target_include_directories(${TARGET_NAME}
        PUBLIC
            "shared"
            "../shared/include"
            "pc")

if(NOT WIN32)
    target_link_libraries(${TARGET_NAME}
            PUBLIC
                Threads::Threads
                ${LIBUSB_LIBRARY})
endif()

target_compile_definitions(${TARGET_NAME}
        PRIVATE
            __PC__
            HAVE_STRUCT_TIMESPEC
            _CRT_SECURE_NO_WARNINGS
        )

if (ENABLE_MYX_PCIE)
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            USE_PCIE)
elseif (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            NO_BOOT
            USE_USB_VSC)
else()
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            USE_USB_VSC)
endif()

