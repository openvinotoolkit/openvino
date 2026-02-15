# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.14)

if(NOT TARGET openvino::runtime)
    find_package(OpenVINO REQUIRED)
endif()

function(add_test)
    set(target "test_${ARGV0}")
    add_executable(${target} "src/${target}.cpp")
    target_link_libraries(${target} openvino::runtime)
    target_compile_definitions(${target} PUBLIC TEST_NAME=${ARGV0})
    install(TARGETS ${target} DESTINATION tests/memory_tests COMPONENT tests EXCLUDE_FROM_ALL)
endfunction()

add_test(inference_sync)
add_test(inference_async)

install(FILES tools/run_tests.py DESTINATION tests/memory_tests/tools COMPONENT tests EXCLUDE_FROM_ALL)