# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Unit tests: graph rewriter (no GPU) and device coordinator (needs 2 GPUs).
add_subdirectory(unit)

# Accuracy of the whole pipeline against a single GPU (needs 2 GPUs).
add_subdirectory(functional)

