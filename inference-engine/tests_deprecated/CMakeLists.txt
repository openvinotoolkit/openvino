# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#####################################################################################################
#                                         SETUP TESTS                                               #
#####################################################################################################

enable_testing()

add_subdirectory(helpers)

if (ENABLE_GAPI_TESTS)
    add_subdirectory(fluid_preproc)
endif()

if (ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

if (ENABLE_BEH_TESTS)
    add_subdirectory(behavior)
endif()

disable_deprecated_warnings()

if(ENABLE_TESTS)
  add_subdirectory(unit)
endif()
