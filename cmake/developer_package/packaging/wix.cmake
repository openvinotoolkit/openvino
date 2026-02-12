# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(ov_wix_specific_settings)
    # installation directory
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Intel")
    # License to be embedded in the installer
    set(CPACK_RESOURCE_FILE_LICENSE "${OpenVINO_SOURCE_DIR}/LICENSE")

    # WIX version (3 for WiX Toolset v3, 4 for WiX .NET Tools)
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
        set(CPACK_WIX_UI_REF "WixUI_FeatureTree")
    endif()

    # Set program menu folder name
    if(NOT DEFINED CPACK_WIX_PROGRAM_MENU_FOLDER)
        set(CPACK_WIX_PROGRAM_MENU_FOLDER "Intel OpenVINO")
    endif()

    # Configure product information
    if(NOT DEFINED CPACK_WIX_ROOT_FEATURE_TITLE)
        set(CPACK_WIX_ROOT_FEATURE_TITLE "Intel(R) Distribution of OpenVINO(TM) Toolkit")
    endif()

    if(NOT DEFINED CPACK_WIX_ROOT_FEATURE_DESCRIPTION)
        set(CPACK_WIX_ROOT_FEATURE_DESCRIPTION "Intel(R) Distribution of OpenVINO(TM) Toolkit ${OpenVINO_VERSION}")
    endif()

    # Set additional WIX properties for Programs and Features
    if(NOT DEFINED CPACK_WIX_PROPERTY_ARPCOMMENTS)
        set(CPACK_WIX_PROPERTY_ARPCOMMENTS "OpenVINO Toolkit for Deep Learning inference")
    endif()

    if(NOT DEFINED CPACK_WIX_PROPERTY_ARPHELPLINK)
        set(CPACK_WIX_PROPERTY_ARPHELPLINK "https://docs.openvino.ai")
    endif()

    if(NOT DEFINED CPACK_WIX_PROPERTY_ARPURLINFOABOUT)
        set(CPACK_WIX_PROPERTY_ARPURLINFOABOUT "https://www.intel.com/openvino")
    endif()

    # Product icon
    # if(EXISTS "${OpenVINO_SOURCE_DIR}/docs/images/openvino-icon.ico")
    #     set(CPACK_WIX_PRODUCT_ICON "${OpenVINO_SOURCE_DIR}/docs/images/openvino-icon.ico")
    # endif()

    # Optional: Custom WIX patch files
    # set(CPACK_WIX_PATCH_FILE "${OpenVINO_SOURCE_DIR}/cmake/packaging/wix_patches.xml")

    # Optional: Extra WIX extensions
    # set(CPACK_WIX_EXTENSIONS "WixUtilExtension")
endmacro()

ov_wix_specific_settings()

#
# ov_archive_cpack_set_dirs()
#
# Set directories for ARCHIVE-style cpack (WIX uses similar structure to archives)
#
macro(ov_archive_cpack_set_dirs)
    # common "archive" package locations
    # TODO: move current variables to OpenVINO specific locations
    set(OV_CPACK_INCLUDEDIR runtime/include)
    set(OV_CPACK_OPENVINO_CMAKEDIR runtime/cmake)
    set(OV_CPACK_DOCDIR docs)
    set(OV_CPACK_LICENSESDIR licenses)
    set(OV_CPACK_SAMPLESDIR samples)
    set(OV_CPACK_WHEELSDIR wheels)
    set(OV_CPACK_DEVREQDIR tools)
    set(OV_CPACK_PYTHONDIR python)

    if(USE_BUILD_TYPE_SUBFOLDER)
        set(build_type ${CMAKE_BUILD_TYPE})
    else()
        set(build_type $<CONFIG>)
    endif()

    if(WIN32)
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_RUNTIMEDIR runtime/bin/${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER}/${build_type})
    elseif(APPLE)
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_RUNTIMEDIR runtime/lib/${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER}/${build_type})
    else()
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_RUNTIMEDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER})
    endif()

    ov_get_pyversion(pyversion)
    if(pyversion)
        set(OV_CPACK_PYTHONDIR python/${pyversion})
    endif()

    # non-runtime package locations
    set(OV_CPACK_TOOLSDIR tools)
    set(OV_CPACK_PLUGINSDIR runtime/lib/${ARCH_FOLDER}/plugins.xml)
endmacro()

ov_archive_cpack_set_dirs()
