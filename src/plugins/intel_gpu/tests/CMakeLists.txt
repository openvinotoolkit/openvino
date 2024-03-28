# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(common)
# Unit tests doesn't link for some reason with icpx
# add_subdirectory(unit)
add_subdirectory(functional)
