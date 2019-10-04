# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "XLink")

include(XLink.cmake)

# Windows threads sources
if(WIN32)
    file(GLOB USB_WIN_SOURCES "../USB_WIN/*")
    set(WIN_PTHREAD_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/../WinPthread/win_semaphore.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/../WinPthread/win_pthread.c")
    list(APPEND XLINK_SOURCES ${USB_WIN_SOURCES} ${WIN_PTHREAD_SOURCES})
else()
    list(APPEND XLINK_SOURCES "../WinPthread/pthread_semaphore.c")
endif()

add_library(${TARGET_NAME} STATIC ${XLINK_SOURCES})

# Threads and usb include
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
        ${XLINK_INCLUDE}
        "../WinPthread")

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
        USE_USB_VSC
        )

if (ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME}
            PRIVATE
            NO_BOOT)
endif()

add_dependencies(${TARGET_NAME} vpu_copy_firmware)

set_property(TARGET ${TARGET_NAME} PROPERTY C_STANDARD 99)
