# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# add dummy `time_tests` target combines all time tests
add_custom_target(time_tests)

# Search OpenVINO Inference Engine installed
find_package(InferenceEngine)
if (NOT InferenceEngine_FOUND)
    # Search OpenVINO Inference Engine via InferenceEngine_DIR
    find_package(IEDevScripts
                 PATHS "${OpenVINO_MAIN_SOURCE_DIR}/cmake/developer_package"
                 NO_CMAKE_FIND_ROOT_PATH
                 NO_DEFAULT_PATH)
    if (NOT InferenceEngine_FOUND)
        # Search OpenVINO Inference Engine via InferenceEngineDeveloperPackage_DIR
        # in order to provide backward compatibility with old OpenVINO packages
        set (HAVE_SYS_STAT_H 1)
        set (HAVE_INTTYPES_H 1)
        set (INTTYPES_FORMAT C99)
        find_package(InferenceEngineDeveloperPackage REQUIRED)
    endif()
endif()

# Build test from every source file.
# Test target name is source file name without extension.
FILE(GLOB tests "*.cpp")

foreach(test_source ${tests})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE IE::inference_engine timetests_helper)

    add_dependencies(time_tests ${test_name})
endforeach()
