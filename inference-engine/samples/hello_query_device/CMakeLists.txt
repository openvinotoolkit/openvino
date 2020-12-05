# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME hello_query_device
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

if(COMMAND ie_add_manifest)
    ie_add_manifest(TARGET_NAME hello_query_device
                    DEPENDENCIES inference_engine)
endif()
