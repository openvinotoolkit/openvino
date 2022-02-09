# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(shared_tests)

if (ENABLE_HDDL OR ENABLE_INTEL_MYRIAD)
    add_subdirectory(vpu)
endif()
