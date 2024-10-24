# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core/dev_api)

add_subdirectory(src)
if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
