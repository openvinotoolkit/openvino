# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var OV_DEVICE_MAPPING BUILD_SHARED_LIBS OV_PLUGINS_HPP_HEADER OV_PLUGINS_HPP_HEADER_IN)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required, but not defined")
    endif()
endforeach()

# configure variables

set(OV_PLUGINS_DECLARATIONS "")
set(OV_PLUGINS_MAP_DEFINITION
    "    static const std::map<Key, Value> plugins_hpp = {")

foreach(dev_map IN LISTS OV_DEVICE_MAPPING)
    string(REPLACE ":" ";" dev_map "${dev_map}")
    list(GET dev_map 0 mapped_dev_name)
    list(GET dev_map 1 actual_dev_name)

    # definitions
    set(dev_config "{")
    if(${mapped_dev_name}_CONFIG)
        string(REPLACE "@" ";" dev_config_parced "${${mapped_dev_name}_CONFIG}")
        foreach(props IN LISTS dev_config_parced)
            string(REPLACE ":" ";" props "${props}")

            list(GET props 0 key)
            list(GET props 1 value)

            set(dev_config "${dev_config} { \"${key}\", \"${value}\" }, ")
        endforeach()
    endif()
    set(dev_config "${dev_config}}")


    if(NOT BUILD_SHARED_LIBS)
        # common
        set(_OV_CREATE_PLUGIN_FUNC "create_plugin_engine_${actual_dev_name}")
        set(_OV_CREATE_EXTENSION_FUNC "create_extensions_${actual_dev_name}")

        # declarations
        set(OV_PLUGINS_DECLARATIONS "${OV_PLUGINS_DECLARATIONS}
        OV_DEFINE_PLUGIN_CREATE_FUNCTION_DECLARATION(${_OV_CREATE_PLUGIN_FUNC});")
        if(${actual_dev_name}_AS_EXTENSION)
            set(OV_PLUGINS_DECLARATIONS "${OV_PLUGINS_DECLARATIONS}
            OV_DEFINE_EXTENSION_CREATE_FUNCTION_DECLARATION(${_OV_CREATE_EXTENSION_FUNC});")
        else()
            set(_OV_CREATE_EXTENSION_FUNC "nullptr")
        endif()

        set(OV_PLUGINS_MAP_DEFINITION "${OV_PLUGINS_MAP_DEFINITION}
        { \"${mapped_dev_name}\", Value { ${_OV_CREATE_PLUGIN_FUNC}, ${_OV_CREATE_EXTENSION_FUNC}, ${dev_config} } },")
    else()
        set(OV_PLUGINS_MAP_DEFINITION "${OV_PLUGINS_MAP_DEFINITION}
        { \"${mapped_dev_name}\", Value { \"${actual_dev_name}\", ${dev_config} } },")
    endif()
endforeach()

set(OV_PLUGINS_MAP_DEFINITION "${OV_PLUGINS_MAP_DEFINITION}
    };\n")

configure_file("${OV_PLUGINS_HPP_HEADER_IN}" "${OV_PLUGINS_HPP_HEADER}" @ONLY)
