# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_tensorflow_frontend_tests")

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE
        gtest_main_manifest frontend_shared_test_classes openvino_tensorflow_frontend
        openvino_tensorflow_frontend_static_tests)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

# Test model generating
ov_check_pip_package(REQUIREMENT tensorflow
                     MESSAGE_MODE WARNING
                     WARNING_MESSAGE "TensorFlow frontend unit tests will be skipped"
                     RESULT_VAR tensorflow_FOUND)

set(TEST_TENSORFLOW_MODELS_DIRNAME test_model_zoo/tensorflow_test_models)
target_compile_definitions(${TARGET_NAME} PRIVATE -D TEST_TENSORFLOW_MODELS_DIRNAME=\"${TEST_TENSORFLOW_MODELS_DIRNAME}/\")

# If 'tensorflow' is not found, code will still be compiled
# but models will not be generated and tests will fail
# This is done this way for 'code style' and check cases - cmake shall pass, but CI machine doesn't need to have
# 'tensorflow' installed to check code style
if (tensorflow_FOUND)
    set(TEST_TENSORFLOW_MODELS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_TENSORFLOW_MODELS_DIRNAME}/)

    file(GLOB_RECURSE TENSORFLOW_GEN_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_scripts/generate_*.py)
    file(GLOB_RECURSE TENSORFLOW_ALL_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
    set(OUT_FILES "")
    foreach(GEN_SCRIPT ${TENSORFLOW_GEN_SCRIPTS})
        get_filename_component(FILE_WE ${GEN_SCRIPT} NAME_WE)
        set(OUT_DONE_FILE ${TEST_TENSORFLOW_MODELS}/${FILE_WE}_done.txt)
        set(OUT_FILES ${OUT_DONE_FILE} ${OUT_FILES})
        add_custom_command(OUTPUT ${OUT_DONE_FILE}
                COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_models/gen_wrapper.py
                ${GEN_SCRIPT}
                ${TEST_TENSORFLOW_MODELS}
                ${OUT_DONE_FILE}
                JOB_POOL four_jobs
                DEPENDS ${TENSORFLOW_ALL_SCRIPTS}
                )
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

add_dependencies(${TARGET_NAME} tensorflow_test_models)

get_target_property(TENSORFLOW_FRONTEND_SRC_DIR openvino_tensorflow_frontend SOURCE_DIR)

add_subdirectory(standalone_build)
add_dependencies(${TARGET_NAME} tensorflow_fe_standalone_build_test)

#
# Install TensorFlow frontend for tests reasons
#

function(ov_frontend_get_file_name target_name library_name)
    set(LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")

    set(LIB_SUFFIX "${IE_BUILD_POSTFIX}")
    if(APPLE)
        set(LIB_SUFFIX "${LIB_SUFFIX}${OpenVINO_VERSION_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    else()
        set(LIB_SUFFIX "${LIB_SUFFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}${OpenVINO_VERSION_SUFFIX}")
    endif()

    set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

set(target_name "${FRONTEND_NAME_PREFIX}tensorflow${FRONTEND_NAME_SUFFIX}")
ov_frontend_get_file_name(${target_name} output_name)

# install with original name for tests component
install(FILES $<TARGET_FILE:${target_name}>
        DESTINATION tests
        COMPONENT tests
        RENAME ${output_name}
        EXCLUDE_FROM_ALL)

install(TARGETS ${target_name}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
