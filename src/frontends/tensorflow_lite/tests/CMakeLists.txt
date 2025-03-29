# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_tensorflow_lite_frontend_tests")

ov_check_pip_packages(REQUIREMENTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
                      MESSAGE_MODE WARNING
                      WARNING_MESSAGE "TensorFlow Lite testing models weren't generated, some tests will fail due models not found"
                      RESULT_VAR tensorflow_FOUND)

if(tensorflow_FOUND)
    set(ctest_labels OV UNIT)
endif()

ov_add_test_target(
    NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            tensorflow_lite_test_models
            tensorflow_lite_fe_standalone_build_test
        LINK_LIBRARIES
            gtest_main_manifest 
            frontend_shared_test_classes 
            openvino_tensorflow_lite_frontend
        ADD_CLANG_FORMAT
        LABELS
            ${ctest_labels} TFL_FE
)

# Test model generating

set(TEST_TENSORFLOW_LITE_MODELS_DIRNAME ${TEST_MODEL_ZOO}/tensorflow_lite_test_models)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_TENSORFLOW_LITE_MODELS_DIRNAME=\"${TEST_TENSORFLOW_LITE_MODELS_DIRNAME}/\")

# If 'tensorflow' is not found, code will still be compiled
# but models will not be generated and tests will fail
# This is done this way for 'code style' and check cases - cmake shall pass, but CI machine doesn't need to have
# 'tensorflow' installed to check code style
if (tensorflow_FOUND)
    set(TEST_TENSORFLOW_LITE_MODELS "${TEST_MODEL_ZOO_OUTPUT_DIR}/tensorflow_lite_test_models")

    file(GLOB_RECURSE TENSORFLOW_GEN_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_scripts/generate_*.py)
    file(GLOB_RECURSE TENSORFLOW_ALL_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
    set(OUT_FILES "")
    foreach(GEN_SCRIPT ${TENSORFLOW_GEN_SCRIPTS})
        get_filename_component(FILE_WE ${GEN_SCRIPT} NAME_WE)
        set(OUT_DONE_FILE ${TEST_TENSORFLOW_LITE_MODELS}/${FILE_WE}_done.txt)
        set(OUT_FILES ${OUT_DONE_FILE} ${OUT_FILES})
        add_custom_command(OUTPUT ${OUT_DONE_FILE}
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_wrapper.py
                    ${GEN_SCRIPT}
                    ${TEST_TENSORFLOW_LITE_MODELS}
                    ${OUT_DONE_FILE}
                JOB_POOL four_jobs
                DEPENDS ${TENSORFLOW_ALL_SCRIPTS})
    endforeach()
    add_custom_target(tensorflow_lite_test_models DEPENDS ${OUT_FILES})

    install(DIRECTORY ${TEST_TENSORFLOW_LITE_MODELS}
            DESTINATION tests/${TEST_TENSORFLOW_LITE_MODELS_DIRNAME}
            COMPONENT tests
            EXCLUDE_FROM_ALL)
else()
    # Produce warning message at build time as well
    add_custom_command(OUTPUT unable_build_tensorflow_models.txt
            COMMAND ${CMAKE_COMMAND}
            -E cmake_echo_color --red "Warning: Unable to generate tensorflow lite test models. Running '${TARGET_NAME}' will likely fail"
            )
    add_custom_target(tensorflow_lite_test_models DEPENDS unable_build_tensorflow_models.txt)
endif()

get_target_property(TENSORFLOW_LITE_FRONTEND_SRC_DIR openvino_tensorflow_lite_frontend SOURCE_DIR)

add_subdirectory(standalone_build)
add_dependencies(${TARGET_NAME} tensorflow_lite_fe_standalone_build_test)
