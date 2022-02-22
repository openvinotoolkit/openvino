# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var OV_FRONTENDS_HPP_HEADER_IN OV_FRONTENDS_HPP_HEADER FRONTEND_NAMES)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is required, but not defined")
    endif()
endforeach()

# configure variables

set(OV_FRONTEND_DECLARATIONS "")
set(OV_FRONTEND_MAP_DEFINITION "    FrontendsStaticRegistry registry = {")

foreach(frontend IN LISTS FRONTEND_NAMES)
    # common
    set(_OV_FRONTEND_DATA_FUNC "GetFrontEndData${frontend}")
    set(_OV_VERSION_FUNC "GetAPIVersion${frontend}")

    # declarations
    set(OV_FRONTEND_DECLARATIONS "${OV_FRONTEND_DECLARATIONS}
ov::frontend::FrontEndVersion ${_OV_VERSION_FUNC}();
void* ${_OV_FRONTEND_DATA_FUNC}();")

    set(OV_FRONTEND_MAP_DEFINITION "${OV_FRONTEND_MAP_DEFINITION}
        { Value { ${_OV_FRONTEND_DATA_FUNC}, ${_OV_VERSION_FUNC} } },")
endforeach()

set(OV_FRONTEND_MAP_DEFINITION "${OV_FRONTEND_MAP_DEFINITION}
    };
    return registry;")

configure_file("${OV_FRONTENDS_HPP_HEADER_IN}" "${OV_FRONTENDS_HPP_HEADER}" @ONLY)
