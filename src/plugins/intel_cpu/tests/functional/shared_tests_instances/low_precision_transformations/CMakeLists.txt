# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(AARCH64)
    add_subdirectory(aarch64)
endif()

if(X86_64)
    add_subdirectory(x64)
endif()
