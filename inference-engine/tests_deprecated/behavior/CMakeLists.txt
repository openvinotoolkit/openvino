# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(shared_tests)

disable_deprecated_warnings()

if (ENABLE_HDDL OR ENABLE_MYRIAD)
    add_subdirectory(vpu)
endif()
