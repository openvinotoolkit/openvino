# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "MvncTests")
set(CMAKE_CXX_STANDARD 11)

set(PRIVATE_INCLUDE
        "utils"
        ${IE_MAIN_SOURCE_DIR}/tests/libs/gtest/googletest/include
        ${IE_MAIN_SOURCE_DIR}/tests/libs/gtest/googletest/
        ${LIBUSB_INCLUDE_DIR}
        "../../shared/include"
        "../../shared"
        "../../XLink/shared"
        "../../watchdog"
        "../../XLink/pc")

file(GLOB_RECURSE SHARED "../../shared/*")

if(ENABLE_MYRIAD_NO_BOOT)
    file(GLOB MVNC_TESTS_SOURCES "mvnc_no_boot_tests.cpp")
else()
    file(GLOB MVNC_TESTS_SOURCES "mvnc_tests_usb.cpp"
                                 "mvnc_tests_common.cpp")
endif()

if(WIN32)
    file(GLOB USB_WIN_SOURCES "../../USB_WIN/*")
    file(GLOB WIN_PTHREAD_SOURCES "../../WinPthread/*")
    list(APPEND ${MVNC_TESTS_SOURCES} ${USB_WIN_SOURCES} ${WIN_PTHREAD_SOURCES})
endif()

add_executable(${TARGET_NAME} ${MVNC_TESTS_SOURCES})
target_include_directories(${TARGET_NAME}
    PUBLIC
        "../include"
    PRIVATE
        ${PRIVATE_INCLUDE})

if(WIN32)
    target_include_directories(${TARGET_NAME}
            PRIVATE
            "../../USB_WIN"
            "../../WinPthread")
endif()

target_compile_definitions(${TARGET_NAME}
        PRIVATE
            __PC__)

target_link_libraries(${TARGET_NAME} gtest gtest_main mvnc)

################# MvncStressTests ###################

set(TARGET_NAME "MvncStressTests")
set(CMAKE_CXX_STANDARD 11)

file(GLOB MVNC_STRESS_TESTS_SOURCES "mvnc_stress_tests.cpp")
add_executable(${TARGET_NAME} ${MVNC_STRESS_TESTS_SOURCES} )

target_include_directories(${TARGET_NAME}
        PUBLIC
            "../include"
        PRIVATE
            ${PRIVATE_INCLUDE})

if(ENABLE_MYRIAD_NO_BOOT)
    target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
endif()


target_link_libraries(${TARGET_NAME} gtest gtest_main mvnc)

################# MvncUtilsTests ###################

set(TARGET_NAME "MvncUtilsTests")
set(CMAKE_CXX_STANDARD 11)

file(GLOB MVNC_UTILS_TESTS_SOURCES "mvnc_utils_tests.cpp")
add_executable(${TARGET_NAME} ${MVNC_UTILS_TESTS_SOURCES} ${SHARED})

target_include_directories(${TARGET_NAME}
        PUBLIC
            "../include"
        PRIVATE
            ${PRIVATE_INCLUDE})


target_link_libraries(${TARGET_NAME} gtest gtest_main mvnc)
