# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(file_content
"<ie>
    <plugins>
    </plugins>
</ie>")

if(NOT EXISTS "${IE_CONFIG_OUTPUT_FILE}")
    file(WRITE "${IE_CONFIG_OUTPUT_FILE}" "${file_content}")
endif()

# get list of plugin files
file(GLOB plugin_files "${IE_CONFIGS_DIR}/*.xml")

function(check_plugin_exists plugin_name outvar)
    set(${outvar} OFF PARENT_SCOPE)

    # check if config file already has this plugin
    file(STRINGS "${IE_CONFIG_OUTPUT_FILE}" content REGEX "plugin .*=\"")

    foreach(line IN LISTS content)
        string(REGEX MATCH "location=\"([^\"]*)\"" location "${line}")
        get_filename_component(location "${CMAKE_MATCH_1}" NAME_WE)

        if("${CMAKE_SHARED_MODULE_PREFIX}${plugin_name}" MATCHES "${location}")
            # plugin has already registered
            set(${outvar} ON PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

set(plugin_files_to_add)
foreach(plugin_file IN LISTS plugin_files)
    get_filename_component(plugin_name "${plugin_file}" NAME_WE)
    check_plugin_exists("${plugin_name}" exists)

    if(NOT exists)
        list(APPEND plugin_files_to_add "${plugin_file}")
    endif()
endforeach()

# add plugin
set(newContent "")
file(STRINGS "${IE_CONFIG_OUTPUT_FILE}" content)

foreach(line IN LISTS content)
    if("${line}" MATCHES "</plugins>")
        foreach(plugin_file IN LISTS plugin_files_to_add)
            file(READ "${plugin_file}" content)
            set(newContent "${newContent}
${content}")
        endforeach()
    endif()

    if(newContent)
        set(newContent "${newContent}\n${line}")
    else()
        set(newContent "${line}")
    endif()
endforeach()

file(WRITE "${IE_CONFIG_OUTPUT_FILE}" "${newContent}")
