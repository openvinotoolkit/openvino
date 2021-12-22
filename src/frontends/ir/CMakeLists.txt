# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME ir
                FILEDESCRIPTION "FrontEnd to load OpenVINO IR file format"
                LINK_LIBRARIES pugixml::static
                               # TODO: remove dependency below in CVS-69781
                               openvino::runtime::dev)

# Add include path to so_extension.hpp
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/frontend.cpp
    PROPERTIES INCLUDE_DIRECTORIES "${OpenVINO_SOURCE_DIR}/src/core/src/")
