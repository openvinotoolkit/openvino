# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0007 NEW)

# OV_PLUGIN_LIBRARY_NAMES is a list. A single library keeps the legacy `location` attribute
# (byte-identical to before); several libraries (a dispatch group) become ordered <location>
# child elements, which ov::Core reconciles per device at runtime.
list(LENGTH OV_PLUGIN_LIBRARY_NAMES num_libraries)
if(num_libraries GREATER 1)
    set(newContent "        <plugin name=\"${OV_DEVICE_NAME}\">")
    foreach(library_name IN LISTS OV_PLUGIN_LIBRARY_NAMES)
        set(newContent "${newContent}
            <location>${library_name}</location>")
    endforeach()
else()
    set(newContent "        <plugin name=\"${OV_DEVICE_NAME}\" location=\"${OV_PLUGIN_LIBRARY_NAMES}\">")
endif()

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
