# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)
set(TARGET_NAME helpers)

file(GLOB HELPERS_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file (GLOB HELPERS_INCLUDES
       ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
      )

## Enable Models multiple search pathes
message("configuring file: ${PROJECT_BINARY_DIR}/test_model_repo.h")
configure_file(test_model_repo.hpp.in ${PROJECT_BINARY_DIR}/test_model_repo.hpp @ONLY)


add_library(${TARGET_NAME} STATIC
            ${HELPERS_SRC}
            ${HELPERS_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_BINARY_DIR})
target_compile_definitions(${TARGET_NAME} PUBLIC -DMODELS_PATH=\"${MODELS_PATH}\")

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_PDB_NAME ${TARGET_NAME})
