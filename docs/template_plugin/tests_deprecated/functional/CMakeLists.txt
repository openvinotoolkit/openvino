# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

disable_deprecated_warnings()

# [cmake:functional_tests]
set(TARGET_NAME TemplateFunctionalTests)

file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${TARGET_NAME} ${TEST_SOURCES})

# link a library with common Inference Engine tests
target_link_libraries(${TARGET_NAME} PRIVATE IE::IESharedTests)

# make sure plugin is built before tests are run
add_dependencies(${TARGET_NAME} templatePlugin)
# [cmake:functional_tests]

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
