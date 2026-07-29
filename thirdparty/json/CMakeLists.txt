# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# suppress shadowing names warning
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)
