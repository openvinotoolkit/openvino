# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

disable_deprecated_warnings()

add_subdirectory(shared_tests)

if (ENABLE_MYRIAD)
    add_subdirectory(vpu)
endif()

if (ENABLE_GNA)
    add_subdirectory(gna)
endif()
