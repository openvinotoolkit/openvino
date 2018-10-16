# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
project( ade )

file( GLOB_RECURSE sources source/*.cpp )
file( GLOB_RECURSE include *.hpp )

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wconversion -Wno-error=cpp" )
endif()

add_library( ${PROJECT_NAME} STATIC ${include} ${sources} )
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties( ${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE True)
