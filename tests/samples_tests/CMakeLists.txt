# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(samples_tests)

install(DIRECTORY smoke_tests/ DESTINATION tests/smoke_tests COMPONENT tests EXCLUDE_FROM_ALL PATTERN "requirements.txt" EXCLUDE)
install(FILES ../constraints.txt DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

file(READ smoke_tests/requirements.txt REQUIREMENTS_REPO)
string(REPLACE "-c ../../constraints.txt" "-c ../constraints.txt" REQUIREMENTS_TMP ${REQUIREMENTS_REPO})
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/smoke_tests/requirements.txt ${REQUIREMENTS_TMP})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/smoke_tests/requirements.txt DESTINATION tests/smoke_tests COMPONENT tests EXCLUDE_FROM_ALL)
