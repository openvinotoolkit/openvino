# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var IE_DEVICE_MAPPING IE_PLUGINS_HPP_HEADER IE_PLUGINS_HPP_HEADER_IN)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required, but not defined")
    endif()
endforeach()

# configure variables

set(IE_PLUGINS_DECLARATIONS "")
set(IE_PLUGINS_MAP_DEFINITION
    "    static const std::map<Key, Value> plugins_hpp = {")

foreach(dev_map IN LISTS IE_DEVICE_MAPPING)
    string(REPLACE ":" ";" dev_map "${dev_map}")
    list(GET dev_map 0 mapped_dev_name)
    list(GET dev_map 1 actual_dev_name)

    # common
    set(_IE_CREATE_PLUGIN_FUNC "CreatePluginEngine${actual_dev_name}")
    set(_IE_CREATE_EXTENSION_FUNC "CreateExtensionShared${actual_dev_name}")

    # declarations
    set(IE_PLUGINS_DECLARATIONS "${IE_PLUGINS_DECLARATIONS}
IE_DEFINE_PLUGIN_CREATE_FUNCTION_DECLARATION(${_IE_CREATE_PLUGIN_FUNC});")
    if(${actual_dev_name}_AS_EXTENSION)
        set(IE_PLUGINS_DECLARATIONS "${IE_PLUGINS_DECLARATIONS}
IE_DEFINE_EXTENSION_CREATE_FUNCTION_DECLARATION(${_IE_CREATE_EXTENSION_FUNC});")
    else()
        set(_IE_CREATE_EXTENSION_FUNC "nullptr")
    endif()

    # definitions
    set(dev_config "{")
    if(${mapped_dev_name}_CONFIG)
        foreach(props IN LISTS ${mapped_dev_name}_CONFIG)
            string(REPLACE ":" ";" props "${props}")

            list(GET props 0 key)
            list(GET props 1 value)

            set(dev_config "${dev_config} { \"${key}\", \"${value}\" }, ")
        endforeach()
    endif()
    set(dev_config "${dev_config}}")

    set(IE_PLUGINS_MAP_DEFINITION "${IE_PLUGINS_MAP_DEFINITION}
        { \"${mapped_dev_name}\", Value { ${_IE_CREATE_PLUGIN_FUNC}, ${_IE_CREATE_EXTENSION_FUNC}, ${dev_config} } },")
endforeach()

set(IE_PLUGINS_MAP_DEFINITION "${IE_PLUGINS_MAP_DEFINITION}
    };\n")

configure_file("${IE_PLUGINS_HPP_HEADER_IN}" "${IE_PLUGINS_HPP_HEADER}" @ONLY)
