# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# add dummy `fuzz` target combines all fuzz tests
add_custom_target(fuzz)

# Build fuzz test from every source file matchs *-fuzzer.cc.
# Fuzz test target name is source file name without extension.
FILE(GLOB tests "*-fuzzer.cc")

add_subdirectory(../../../thirdparty/cnpy ${CMAKE_CURRENT_BINARY_DIR}/cnpy)
add_subdirectory(../../../thirdparty/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)

if(OV_LIBASAN_FILEPATH)
    message(STATUS "For tests pre-load asan library used for OpenVINO build:")
    message(STATUS "    LD_PRELOAD=${OV_LIBASAN_FILEPATH} [fuzzer-executable]")
endif()

foreach(test_source ${tests})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_fuzzer(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE
        openvino::runtime openvino::cnpy openvino::zlib)

    add_dependencies(fuzz ${test_name})

    install(TARGETS ${test_name}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endforeach()
