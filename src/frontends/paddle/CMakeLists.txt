# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(src)

# ticket 95904
if(ENABLE_TESTS)
   add_subdirectory(tests)
endif()
