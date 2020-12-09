# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_CONFORMANCE_TESTS)
    add_subdirectory(subgraphs_dumper)
    add_subdirectory(subgraphs_dumper/tests)
    add_subdirectory(test_runner)
endif()