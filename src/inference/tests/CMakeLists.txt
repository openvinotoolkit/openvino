# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Close LTO temporarily for add_symbol_to_partition_1 issue
set(ENABLE_LTO OFF)

add_subdirectory(unit)
add_subdirectory(functional)
