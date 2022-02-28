# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_CXX_STANDARD 11)

include(${XLINK_DIR}/XLink.cmake)

set(MVNC_TESTS_COMMON_INCLUDE
        "cases"
        "helpers"
        )

file(GLOB MVNC_TESTS_COMMON_SOURCES
        "helpers/mvnc_test_helper.cpp"
        "cases/mvnc_common_test_cases.cpp")

set(PRIVATE_INCLUDE
        ${XLINK_INCLUDE}
        ${XLINK_PLATFORM_INCLUDE}
        ${WATCHDOG_INCLUDE})

function(add_mvnc_test_target TARGET_NAME TESTS_SOURCES)
    add_executable(${TARGET_NAME} ${TESTS_SOURCES} ${MVNC_TESTS_COMMON_SOURCES})

    target_include_directories(${TARGET_NAME}
            PUBLIC
            "../include"
            PRIVATE
            ${MVNC_TESTS_COMMON_INCLUDE}
            ${PRIVATE_INCLUDE})

    if(ENABLE_MYRIAD_NO_BOOT)
        target_compile_definitions(${TARGET_NAME} PRIVATE NO_BOOT)
    endif()

    target_compile_definitions(${TARGET_NAME}
            PRIVATE
            __PC__)

    target_link_libraries(${TARGET_NAME} gtest gtest_main mvnc)
    add_dependencies(${TARGET_NAME} vpu_copy_firmware)
endfunction()

################# MvncTests ###################

if(ENABLE_MYRIAD_NO_BOOT)
    set(MVNC_TESTS_SOURCES
            mvnc_no_boot_tests.cpp
            cases/mvnc_no_boot_test_cases.cpp
            )
else()
    set(MVNC_TESTS_SOURCES
            mvnc_tests_common.cpp
            mvnc_tests_usb.cpp
            cases/mvnc_usb_test_cases.cpp)
endif()

add_mvnc_test_target("MvncTests" "${MVNC_TESTS_SOURCES}")

################# MvncStressTests ###################

set(MVNC_STRESS_TESTS_SOURCES
        mvnc_stress_tests.cpp
        cases/mvnc_stress_test_cases.cpp
        )
add_mvnc_test_target("MvncStressTests" "${MVNC_STRESS_TESTS_SOURCES}")

################# MvncUtilsTests ###################

set(MVNC_UTILS_TESTS_SOURCES mvnc_utils_tests.cpp)
add_mvnc_test_target("MvncUtilsTests" "${MVNC_UTILS_TESTS_SOURCES}")
