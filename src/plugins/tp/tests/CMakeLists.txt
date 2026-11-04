# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Unit tests: the L1 transformations (no GPU).
add_subdirectory(unit)

# Accuracy of the whole pipeline against a single GPU (needs 2-4 GPUs).
add_subdirectory(functional)

