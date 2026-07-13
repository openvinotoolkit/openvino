# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_tensorflow_frontend_tests")

list(APPEND TF_TESTS_DEPENDENCIES tensorflow_test_models tensorflow_fe_standalone_build_test openvino_intel_cpu_plugin)
if (NOT ENABLE_INTEL_CPU)
    list(REMOVE_ITEM TF_TESTS_DEPENDENCIES openvino_intel_cpu_plugin)
    set(EXCLUDED_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/compilation.cpp)
endif()

ov_check_pip_packages(REQUIREMENTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
                      MESSAGE_MODE WARNING
                      WARNING_MESSAGE "TensorFlow testing models weren't generated, some tests will fail due models not found"
                      RESULT_VAR tensorflow_FOUND)

if(tensorflow_FOUND)
    set(ctest_labels OV UNIT)
endif()

ov_add_test_target(
    NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_TESTS}
        DEPENDENCIES ${TF_TESTS_DEPENDENCIES}
        LINK_LIBRARIES
            gtest_main_manifest
            frontend_shared_test_classes
            openvino_tensorflow_frontend
            openvino_tensorflow_common
        ADD_CLANG_FORMAT
        LABELS
            ${ctest_labels} TF_FE
)

# Test model generating

set(TEST_TENSORFLOW_MODELS_DIRNAME ${TEST_MODEL_ZOO}/tensorflow_test_models)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_TENSORFLOW_MODELS_DIRNAME=\"${TEST_TENSORFLOW_MODELS_DIRNAME}/\")

# If 'tensorflow' is not found, code will still be compiled
# but models will not be generated and tests will fail
# This is done this way for 'code style' and check cases - cmake shall pass, but CI machine doesn't need to have
# 'tensorflow' installed to check code style
if (tensorflow_FOUND)
    set(TEST_TENSORFLOW_MODELS ${TEST_MODEL_ZOO_OUTPUT_DIR}/tensorflow_test_models/)

    file(GLOB_RECURSE TENSORFLOW_GEN_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_scripts/generate_*.py)
    file(GLOB_RECURSE TENSORFLOW_MODELS_PBTXT ${CMAKE_CURRENT_SOURCE_DIR}/test_models/models_pbtxt/*.pbtxt)
    list (APPEND TENSORFLOW_GEN_SCRIPTS ${TENSORFLOW_MODELS_PBTXT})
    file(GLOB_RECURSE TENSORFLOW_ALL_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
    set(OUT_FILES "")
    foreach(GEN_SCRIPT ${TENSORFLOW_GEN_SCRIPTS})
        get_filename_component(FILE_WE ${GEN_SCRIPT} NAME_WE)
        set(OUT_DONE_FILE ${TEST_TENSORFLOW_MODELS}/${FILE_WE}_done.txt)
        set(OUT_FILES ${OUT_DONE_FILE} ${OUT_FILES})
        add_custom_command(OUTPUT ${OUT_DONE_FILE}
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_wrapper.py
                    ${GEN_SCRIPT}
                    ${TEST_TENSORFLOW_MODELS}
                    ${OUT_DONE_FILE}
                JOB_POOL four_jobs
                DEPENDS ${TENSORFLOW_ALL_SCRIPTS})
    endforeach()
    add_custom_target(tensorflow_test_models DEPENDS ${OUT_FILES})

    install(DIRECTORY ${TEST_TENSORFLOW_MODELS}
            DESTINATION tests/${TEST_TENSORFLOW_MODELS_DIRNAME}
            COMPONENT tests
            EXCLUDE_FROM_ALL)
else()
    # Produce warning message at build time as well
    add_custom_command(OUTPUT unable_build_tensorflow_models.txt
            COMMAND ${CMAKE_COMMAND}
            -E cmake_echo_color --red "Warning: Unable to generate tensorflow test models. Running '${TARGET_NAME}' will likely fail"
            )
    add_custom_target(tensorflow_test_models DEPENDS unable_build_tensorflow_models.txt)
endif()

get_target_property(TENSORFLOW_FRONTEND_SRC_DIR openvino_tensorflow_frontend SOURCE_DIR)

# switch_marker_lifetime.cpp reaches into the frontend-internal conditional-flow marker
# (CfMarkerType in src/tf_utils.hpp, pulled via helper_ops/switch.hpp) to assert a Switch node is
# not kept alive by its own rt_info. That needs the frontend 'src' dir, the generated protobuf
# headers (frontend binary dir) and the protobuf include on the path. These are added target-wide;
# existing tests are unaffected because a quoted #include "tf_utils.hpp" resolves to the sibling
# test wrapper (tests/tf_utils.hpp) first, while helper_ops/switch.hpp has no sibling and so falls
# back to the frontend 'src' copy.
target_include_directories(${TARGET_NAME} PRIVATE
    ${TENSORFLOW_FRONTEND_SRC_DIR}
    $<TARGET_PROPERTY:openvino_tensorflow_frontend,BINARY_DIR>
    $<TARGET_PROPERTY:libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>)

ov_build_target_faster(${TARGET_NAME} PCH)

add_subdirectory(standalone_build)
add_dependencies(${TARGET_NAME} tensorflow_fe_standalone_build_test)
