# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

disable_deprecated_warnings()

add_subdirectory(shared_tests)

if (ENABLE_INTEL_MYRIAD)
    add_subdirectory(vpu)
endif()

if (ENABLE_INTEL_GNA)
    add_subdirectory(gna)
endif()
