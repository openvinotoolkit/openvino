# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#####################################################################################################
#                                         SETUP TESTS                                               #
#####################################################################################################

enable_testing()

add_subdirectory(mock_engine)

add_subdirectory(helpers)

disable_deprecated_warnings()

if(ENABLE_TESTS)
  add_subdirectory(unit)
endif()
