# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

disable_deprecated_warnings()

function(enable_vpu TARGET_NAME FLAG_NAME PLUGIN_NAME)
    # Common tests for HDDL MYRIAD KMB
    file(GLOB_RECURSE TEST_INCLUDE
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/*.hpp)

    file(GLOB_RECURSE TEST_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/*.cpp
            )

    # Tests for Myriad plugin
    if (ENABLE_MYRIAD)
        file(GLOB_RECURSE TEST_INCLUDE myriad_tests/*.hpp)
        file(GLOB_RECURSE VPU_TESTS myriad_tests/*.cpp)
        if (NOT ENABLE_MYRIAD_NO_BOOT)
            list(REMOVE_ITEM VPU_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/myriad_tests/vpu_boot_tests.cpp)
        endif()
        list(APPEND TEST_SRC ${VPU_TESTS})
    endif()

    # Tests for HDDL plugin
    if (ENABLE_HDDL)
        file(GLOB HDDL_TESTS hddl_tests/*.cpp)
        list(APPEND TEST_SRC ${HDDL_TESTS})
    endif()

    list(APPEND DEPENDENCIES
            ${PLUGIN_NAME}
            vpu_copy_firmware)

    source_group("src" FILES ${TEST_SRC})
    source_group("include" FILES ${TEST_INCLUDE})

    add_executable(${TARGET_NAME}
            ${TEST_SRC}
            ${TEST_INCLUDE})

    target_compile_definitions(${TARGET_NAME} PRIVATE
            ${FLAG_NAME}=1)

    target_link_libraries(${TARGET_NAME} PRIVATE
            IEBehaviorSharedTests)

    target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/plugin_tests)

    if (ENABLE_MYRIAD)
        include(${XLINK_DIR}/XLink.cmake)

        target_compile_definitions(${TARGET_NAME} PRIVATE __PC__)

        target_include_directories(${TARGET_NAME} PRIVATE
                ${IE_MAIN_SOURCE_DIR}/src/vpu/common/include
                ${CMAKE_CURRENT_SOURCE_DIR}/myriad_tests
                ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius
                ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include
                ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include/watchdog
                ${XLINK_INCLUDE}
                ${XLINK_PLATFORM_INCLUDE})
        target_link_libraries(${TARGET_NAME} PRIVATE mvnc)
    endif()

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
    set_property(TEST ${TARGET_NAME} PROPERTY LABELS VPU MYRIAD)

    add_dependencies(${TARGET_NAME} ${DEPENDENCIES})
endfunction(enable_vpu)

if (ENABLE_MYRIAD)
    set(MYRIAD_TARGET_NAME MyriadBehaviorTests)
    enable_vpu(${MYRIAD_TARGET_NAME} USE_MYRIAD myriadPlugin)
endif()
