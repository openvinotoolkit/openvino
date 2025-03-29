# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0007 NEW)

set(newContent "        <plugin name=\"${OV_DEVICE_NAME}\" location=\"${OV_PLUGIN_LIBRARY_NAME}\">")

if(OV_PLUGIN_PROPERTIES)
    set(newContent "${newContent}
            <properties>")

    foreach(props IN LISTS OV_PLUGIN_PROPERTIES)
        string(REPLACE ":" ";" props "${props}")

        list(GET props 0 key)
        list(GET props 1 value)

        set(newContent "${newContent}
                <property key=\"${key}\" value=\"${value}\"/>")
    endforeach()

    set(newContent "${newContent}
            </properties>")
endif()

set(newContent "${newContent}
        </plugin>")

file(WRITE "${OV_CONFIG_OUTPUT_FILE}" "${newContent}")
