# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "tensorflow_tests")

file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(FILTER SRC EXCLUDE REGEX standalone_build)

add_executable(${TARGET_NAME} ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE frontend_shared_test_classes openvino_tensorflow_frontend openvino::runtime)

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

# Test model generating
ie_check_pip_package(tensorflow WARNING)

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
add_dependencies(${TARGET_NAME} openvino_tensorflow_frontend)

get_target_property(TENSORFLOW_FRONTEND_SRC_DIR openvino_tensorflow_frontend SOURCE_DIR)
target_include_directories(${TARGET_NAME} PRIVATE ${TENSORFLOW_FRONTEND_SRC_DIR}/src/pass/)

add_subdirectory(standalone_build)
add_dependencies(${TARGET_NAME} tensorflow_fe_standalone_build_test)
