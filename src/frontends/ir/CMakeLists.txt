# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME ir
                FILEDESCRIPTION "FrontEnd to load OpenVINO IR file format"
                LINK_LIBRARIES openvino::pugixml
                               # TODO: remove dependency below in CVS-69781
                               openvino::runtime::dev)
