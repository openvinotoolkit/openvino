# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB local_modules "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(module_path IN LISTS local_modules)
    if( # Skip files
        NOT IS_DIRECTORY ${module_path} OR
        # or directories without cmake
        NOT EXISTS "${module_path}/CMakeLists.txt")
        continue()
    endif()
    add_subdirectory(${module_path})
endforeach()
