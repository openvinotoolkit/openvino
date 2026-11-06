# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Unit tests: the L1 transformations (no device).
add_subdirectory(unit)

# The TP plugin end to end, its ranks on 2-4 devices of type OV_TP_TEST_DEVICE (GPU by default).
# The collective contract the rank devices meet is tested by the device plugins
# (src/tests/functional/plugin/shared/include/behavior/tp/).
add_subdirectory(functional)

