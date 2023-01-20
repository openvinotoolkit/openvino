# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(unit)
add_subdirectory(functional)

if(ENABLE_GAPI_TESTS)
    add_subdirectory(fluid_preproc)
endif()
