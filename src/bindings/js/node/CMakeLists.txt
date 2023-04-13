# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#add_subdirectory(src)

execute_process(COMMAND npm --prefix ${OpenVINO_SOURCE_DIR}/src/bindings/js/node-addon install)
