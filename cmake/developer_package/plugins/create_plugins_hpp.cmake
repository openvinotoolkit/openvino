# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var IE_DEVICE_NAMES IE_PLUGINS_HPP_HEADER IE_PLUGINS_HPP_HEADER_IN)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required, but not defined")
    endif()
endforeach()

# configure variables

set(IE_PLUGINS_DECLARATIONS "")
set(IE_PLUGINS_MAP_DEFINITION
    "std::map<std::string, InferenceEngine::CreatePluginEngineFunc *> plugins_hpp = {")

foreach(dev_name IN LISTS IE_DEVICE_NAMES)
    set(_IE_CREATE_PLUGIN_FUNC "CreatePluginEngine${dev_name}")
    set(IE_PLUGINS_DECLARATIONS "${IE_PLUGINS_DECLARATIONS}
IE_DEFINE_PLUGIN_CREATE_FUNCTION_DECLARATION(${_IE_CREATE_PLUGIN_FUNC})")
    set(IE_PLUGINS_MAP_DEFINITION "${IE_PLUGINS_MAP_DEFINITION}
    { \"${dev_name}\", ${_IE_CREATE_PLUGIN_FUNC} },")
endforeach()

set(IE_PLUGINS_MAP_DEFINITION "${IE_PLUGINS_MAP_DEFINITION}
};\n")


message("${IE_PLUGINS_DECLARATIONS}")
message("${IE_PLUGINS_MAP_DEFINITION}")

configure_file("${IE_PLUGINS_HPP_HEADER_IN}" "${IE_PLUGINS_HPP_HEADER}" @ONLY)
