# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${CMAKE_CURRENT_LIST_DIR}/../developer_package/packaging/license_utils.cmake)

# Components

macro(ov_cpack_settings)
    # fill a list of components which are part of WIX installer
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item IN LISTS cpack_components_all)
        # filter out some components, which are not needed to be wrapped to Windows MSI package
        if(# python wheels are not needed to be wrapped by WIX installer
           NOT item STREQUAL OV_CPACK_COMP_PYTHON_WHEELS AND
           # It was decided not to distribute JAX as C++ component
           NOT item STREQUAL "jax")
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    unset(cpack_components_all)

    # restore the components settings

    foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
        cpack_add_component(${comp} ${_${comp}_cpack_component_args})
    endforeach()

    # WIX specific settings

    # Set WIX version (3 for WiX Toolset v3, 4 for WiX .NET Tools)
    if(NOT DEFINED CPACK_WIX_VERSION)
        set(CPACK_WIX_VERSION "3")
    endif()

    # Set target architecture
    if(NOT DEFINED CPACK_WIX_ARCHITECTURE)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(CPACK_WIX_ARCHITECTURE "x64")
        else()
            set(CPACK_WIX_ARCHITECTURE "x86")
        endif()
    endif()

    # Set install scope to perMachine for system-wide installation
    if(NOT DEFINED CPACK_WIX_INSTALL_SCOPE)
        set(CPACK_WIX_INSTALL_SCOPE "perMachine")
    endif()

    # Set UI reference for WIX installer
    if(NOT DEFINED CPACK_WIX_UI_REF)
        if(CPACK_COMPONENTS_ALL)
            set(CPACK_WIX_UI_REF "WixUI_FeatureTree")
        else()
            set(CPACK_WIX_UI_REF "WixUI_InstallDir")
        endif()
    endif()

    # Set program menu folder name
    if(NOT DEFINED CPACK_WIX_PROGRAM_MENU_FOLDER)
        set(CPACK_WIX_PROGRAM_MENU_FOLDER "${CPACK_PACKAGE_NAME}")
    endif()

    # Configure product information
    if(NOT DEFINED CPACK_WIX_ROOT_FEATURE_TITLE)
        set(CPACK_WIX_ROOT_FEATURE_TITLE "${CPACK_PACKAGE_NAME}")
    endif()

    if(NOT DEFINED CPACK_WIX_ROOT_FEATURE_DESCRIPTION)
        set(CPACK_WIX_ROOT_FEATURE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
    endif()

    # Set product and upgrade GUIDs if needed
    # Note: These should be set in the main CMakeLists.txt for consistency
    # CPACK_WIX_UPGRADE_GUID should be constant across versions to enable upgrades
    # CPACK_WIX_PRODUCT_GUID should be unique per version

    # override package file name
    set(CPACK_PACKAGE_FILE_NAME "openvino_toolkit_${OpenVINO_VERSION}.${OpenVINO_VERSION_BUILD}_installer")

    # Set additional WIX properties for Programs and Features
    if(NOT DEFINED CPACK_WIX_PROPERTY_ARPCOMMENTS)
        set(CPACK_WIX_PROPERTY_ARPCOMMENTS "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
    endif()

    if(NOT DEFINED CPACK_WIX_PROPERTY_ARPHELPLINK)
        set(CPACK_WIX_PROPERTY_ARPHELPLINK "https://docs.openvino.ai")
    endif()

    if(NOT DEFINED CPACK_WIX_PROPERTY_ARPURLINFOABOUT)
        set(CPACK_WIX_PROPERTY_ARPURLINFOABOUT "https://www.intel.com/openvino")
    endif()

    # License file - generate RTF from plain text LICENSE
    if(NOT DEFINED CPACK_RESOURCE_FILE_LICENSE)
        set(_license_source "${OpenVINO_SOURCE_DIR}/LICENSE")
        set(_license_rtf "${CMAKE_BINARY_DIR}/LICENSE.rtf")
        ov_generate_rtf_license("${_license_source}" "${_license_rtf}")
        
        if(EXISTS "${_license_rtf}")
            set(CPACK_RESOURCE_FILE_LICENSE "${_license_rtf}")
        endif()
    endif()
    
    # Set WIX license RTF if license is provided
    if(DEFINED CPACK_RESOURCE_FILE_LICENSE)
        if(CPACK_RESOURCE_FILE_LICENSE MATCHES "\\.rtf$")
            set(CPACK_WIX_LICENSE_RTF "${CPACK_RESOURCE_FILE_LICENSE}")
        endif()
    endif()
endmacro()
