# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Only conditional recursion for architecture-specific subdirectories
if(X86_64)
  add_subdirectory(x64)
elseif(ARM OR AARCH64)
  add_subdirectory(aarch64)
endif()