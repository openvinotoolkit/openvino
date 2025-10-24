# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Recurse to child directories
add_subdirectory(common)

if(X86 OR X86_64)
  add_subdirectory(x64)
endif()

if(ARM OR AARCH64)
  add_subdirectory(arm)
endif()

if(RISCV64)
  add_subdirectory(riscv64)
endif()