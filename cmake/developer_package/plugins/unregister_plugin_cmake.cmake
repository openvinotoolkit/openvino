# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT EXISTS "${OV_CONFIG_OUTPUT_FILE}")
    return()
endif()

# remove plugin file
file(REMOVE "${OV_CONFIGS_DIR}/${OV_PLUGIN_NAME}.xml")

# remove plugin
set(newContent "")
file(STRINGS "${OV_CONFIG_OUTPUT_FILE}" content)

set(skip_plugin OFF)
foreach(line IN LISTS content)
    if("${line}" MATCHES "name=\"${OV_PLUGIN_NAME}\"")
        set(skip_plugin ON)
    endif()

    if(NOT skip_plugin)
        if(newContent)
            set(newContent "${newContent}\n${line}")
        else()
            set(newContent "${line}")
        endif()
    endif()

    if("${line}" MATCHES "</plugin>")
        set(skip_plugin OFF)
    endif()
endforeach()

file(WRITE "${OV_CONFIG_OUTPUT_FILE}" "${newContent}")
