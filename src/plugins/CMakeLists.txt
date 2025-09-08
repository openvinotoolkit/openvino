# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_deprecated_no_errors()

set(skip_module "${CMAKE_CURRENT_SOURCE_DIR}/template")
file(GLOB local_modules "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(module_path IN LISTS local_modules)
    if( # Skip files
        NOT IS_DIRECTORY ${module_path} OR
        # or directories without cmake
        NOT EXISTS "${module_path}/CMakeLists.txt" OR
        # module in the skip list
        ${module_path} STREQUAL ${skip_module})
        continue()
    endif()
    add_subdirectory(${module_path})
endforeach()
