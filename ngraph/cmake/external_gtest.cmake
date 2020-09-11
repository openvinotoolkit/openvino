# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install GoogleTest ...
#------------------------------------------------------------------------------

SET(GTEST_GIT_REPO_URL https://github.com/google/googletest.git)
SET(GTEST_GIT_LABEL release-1.8.1)

set(GMOCK_OUTPUT_DIR ${EXTERNAL_PROJECTS_ROOT}/gtest/build/googlemock)
set(GTEST_OUTPUT_DIR ${GMOCK_OUTPUT_DIR}/gtest)

if(WIN32)
    list(APPEND GTEST_CMAKE_ARGS
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE=${GTEST_OUTPUT_DIR}
        -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG=${GTEST_OUTPUT_DIR}
        -Dgtest_force_shared_crt=TRUE
    )
    set(GMOCK_OUTPUT_DIR ${GTEST_OUTPUT_DIR})
endif()

if(CMAKE_BUILD_TYPE)
    list(APPEND GTEST_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    )
endif()

if(UNIX)
    # workaround for compile error
    # related: https://github.com/intel/mkl-dnn/issues/55
    set(GTEST_CXX_FLAGS "-Wno-unused-result ${CMAKE_ORIGINAL_CXX_FLAGS} -Wno-undef")
else()
    set(GTEST_CXX_FLAGS ${CMAKE_ORIGINAL_CXX_FLAGS})
endif()

#Build for ninja
SET(GTEST_PATHS ${GTEST_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtestd${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${GMOCK_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gmockd${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${GTEST_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${GMOCK_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(
    ext_gtest
    PREFIX gtest
    GIT_REPOSITORY ${GTEST_GIT_REPO_URL}
    GIT_TAG ${GTEST_GIT_LABEL}
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
        ${NGRAPH_FORWARD_CMAKE_ARGS}
        -DCMAKE_CXX_FLAGS=${GTEST_CXX_FLAGS}
        ${GTEST_CMAKE_ARGS}
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/gtest/build"
    EXCLUDE_FROM_ALL TRUE
    BUILD_BYPRODUCTS ${GTEST_PATHS}
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gtest SOURCE_DIR BINARY_DIR)

add_library(libgtest INTERFACE)
add_dependencies(libgtest ext_gtest ext_gmock)
target_include_directories(libgtest SYSTEM INTERFACE
    ${SOURCE_DIR}/googletest/include
    ${SOURCE_DIR}/googlemock/include)

if(LINUX OR APPLE OR WIN32)
    target_link_libraries(libgtest INTERFACE
        debug ${GTEST_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtestd${CMAKE_STATIC_LIBRARY_SUFFIX}
        debug ${GMOCK_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gmockd${CMAKE_STATIC_LIBRARY_SUFFIX}
        optimized ${GTEST_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
        optimized ${GMOCK_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gmock${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    message(FATAL_ERROR "libgtest: Unsupported platform.")
endif()
