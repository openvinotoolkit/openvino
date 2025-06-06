# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

set(ABSL_ENABLE_INSTALL ON)

set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
    abseil-cpp
    EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG d7aaad83b488fd62bd51c81ecf16cd938532cc0a # LTS_2024_01_16
)
FetchContent_MakeAvailable(abseil-cpp)
