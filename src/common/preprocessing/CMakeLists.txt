# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_deprecated_no_errors()

add_subdirectory(src)

if(ENABLE_TESTS AND ENABLE_GAPI_PREPROCESSING)
    add_subdirectory(tests)
endif()
