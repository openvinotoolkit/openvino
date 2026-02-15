# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CPACK_GENERATOR STREQUAL "DEB")
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/debian.cmake")
elseif(CPACK_GENERATOR STREQUAL "NPM")
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/npm.cmake")
elseif(CPACK_GENERATOR STREQUAL "RPM")
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/rpm.cmake")
elseif(CPACK_GENERATOR MATCHES "^(CONDA-FORGE|BREW|CONAN|VCPKG)$")
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/common-libraries.cmake")
elseif(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|TZST|ZIP)$")
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/archive.cmake")
elseif(CPACK_GENERATOR STREQUAL "NSIS")
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/nsis.cmake")
endif()
