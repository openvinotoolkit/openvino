# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

####################################
## All next project will use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################


set (gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")

add_subdirectory(mock_engine)

#####################################################################################################
#                                     SETUP GOOGLE TESTS                                            #
#####################################################################################################

add_subdirectory(libs/gtest)

#####################################################################################################
#                                     SETUP GOOGLE TESTS                                            #
#####################################################################################################
enable_testing()


add_subdirectory(helpers)

disable_deprecated_warnings()

if(ENABLE_TESTS)
  add_subdirectory(unit)
endif()
