# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CPACK_GENERATOR STREQUAL "DEB")
    include(cmake/packaging/debian.cmake)
elseif(CPACK_GENERATOR STREQUAL "RPM")
    include(cmake/packaging/rpm.cmake)
elseif(CPACK_GENERATOR MATCHES "^(CONDA-FORGE|BREW)$")
    include(cmake/packaging/common-libraries.cmake)
elseif(CPACK_GENERATOR STREQUAL "NSIS")
    include(cmake/packaging/nsis.cmake)
endif()
