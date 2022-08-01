# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(shared)
add_subdirectory(abc)
add_subdirectory(bde)

# Use the same code for proxy plugin registration in plugins.xml
#
# ov_add_plugin(NAME PROXY
#               DEVICE_NAME "MOCK"
#               DEFAULT_CONFIG
#                   FALLBACK_PRIORITY:ABC,BDE)
