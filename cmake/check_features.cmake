# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE  ON)
endif()

# FIXME: there are compiler failures with LTO and Cross-Compile toolchains. Disabling for now, but
#        this must be addressed in a proper way
if(CMAKE_CROSSCOMPILING OR NOT LINUX)
    set(ENABLE_LTO OFF)
endif()


print_enabled_features()
