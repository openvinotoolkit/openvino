# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(common)

# Use priority-based architecture selection to avoid multiple definitions
if(X86_64 OR X86)
    add_subdirectory(x64)
elseif(AARCH64 OR ARM)
    add_subdirectory(arm)
elseif(RISCV64)
    add_subdirectory(riscv64)
endif()