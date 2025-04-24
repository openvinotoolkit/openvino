# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_VS_VER_FILEVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
set(OV_VS_VER_PRODUCTVERSION_QUAD "${OpenVINO_VERSION_MAJOR},${OpenVINO_VERSION_MINOR},${OpenVINO_VERSION_PATCH},${OpenVINO_VERSION_BUILD}")
set(OV_VS_VER_FILEVERSION_STR "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}.${OpenVINO_VERSION_BUILD}")

set(OV_VS_VER_COMPANY_NAME_STR "Intel Corporation")
set(OV_VS_VER_PRODUCTVERSION_STR "${CI_BUILD_NUMBER}")
set(OV_VS_VER_PRODUCTNAME_STR "OpenVINO toolkit")
set(OV_VS_VER_COPYRIGHT_STR "Copyright (C) 2018-2021, Intel Corporation")
set(OV_VS_VER_COMMENTS_STR "https://docs.openvino.ai/")

#
# ov_add_vs_version_file(NAME <name>
#                        FILEDESCRIPTION <file description>
#                        [COMPANY_NAME <company name>]
#                        [FILEVERSION <file version>]
#                        [INTERNALNAME <internal name>]
#                        [COPYRIGHT <name>]
#                        [PRODUCTNAME <name>]
#                        [PRODUCTVERSION <name>]
#                        [COMMENTS <name>]
#                        [FILEVERSION_QUAD <name>]
#                        [PRODUCTVERSION_QUAD <name>])
#
function(ov_add_vs_version_file)
    if(NOT WIN32 OR NOT BUILD_SHARED_LIBS)
        return()
    endif()

    cmake_parse_arguments(VS_VER "" "COMPANY_NAME;NAME;FILEDESCRIPTION;FILEVERSION;INTERNALNAME;COPYRIGHT;PRODUCTNAME;PRODUCTVERSION;COMMENTS;FILEVERSION_QUAD;PRODUCTVERSION_QUAD" "" ${ARGN})

    if(NOT TARGET ${VS_VER_NAME})
        message(FATAL_ERROR "${VS_VER_NAME} must define a target")
    endif()

    get_target_property(target_type ${VS_VER_NAME} TYPE)
    if(NOT target_type MATCHES "^(SHARED|MODULE)_LIBRARY$")
        message(FATAL_ERROR "ov_add_vs_version_file can work only with dynamic libraries")
    endif()

    macro(_vs_ver_update_variable name)
        if(VS_VER_NAME AND DEFINED OV_${VS_VER_NAME}_VS_VER_${name})
            set(OV_VS_VER_${name} "${OV_${VS_VER_NAME}_VS_VER_${name}}")
        elseif(VS_VER_${name})
            set(OV_VS_VER_${name} "${VS_VER_${name}}")
        endif()
    endmacro()

    _vs_ver_update_variable(FILEVERSION_QUAD)
    _vs_ver_update_variable(PRODUCTVERSION_QUAD)

    macro(_vs_ver_update_str_variable name)
        if(VS_VER_NAME AND DEFINED OV_${VS_VER_NAME}_VS_VER_${name})
            set(OV_VS_VER_${name}_STR "${OV_${VS_VER_NAME}_VS_VER_${name}}")
        elseif(VS_VER_${name})
            set(OV_VS_VER_${name}_STR "${VS_VER_${name}}")
        endif()
    endmacro()

    _vs_ver_update_str_variable(COMPANY_NAME)
    _vs_ver_update_str_variable(FILEDESCRIPTION)
    _vs_ver_update_str_variable(FILEVERSION)
    _vs_ver_update_str_variable(INTERNALNAME)
    _vs_ver_update_str_variable(COPYRIGHT)
    _vs_ver_update_str_variable(PRODUCTNAME)
    _vs_ver_update_str_variable(PRODUCTVERSION)
    _vs_ver_update_str_variable(COMMENTS)

    set(OV_VS_VER_ORIGINALFILENAME_STR "${CMAKE_SHARED_LIBRARY_PREFIX}${VS_VER_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(OV_VS_VER_INTERNALNAME_STR ${VS_VER_NAME})

    set(vs_version_output "${CMAKE_CURRENT_BINARY_DIR}/vs_version.rc")
    configure_file("${OpenVINODeveloperScripts_DIR}/vs_version/vs_version.rc.in" "${vs_version_output}" @ONLY)

    source_group("src" FILES ${vs_version_output})
    target_sources(${VS_VER_NAME} PRIVATE ${vs_version_output})
endfunction()
