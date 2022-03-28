# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_TBBBIND_2_5)
    # try to find prebuilt version of tbbbind_2_5
    find_package(TBBBIND_2_5 QUIET)
    if(TBBBIND_2_5_FOUND)
        message(STATUS "Static tbbbind_2_5 package is found")
        set_target_properties(${TBBBIND_2_5_IMPORTED_TARGETS} PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS TBBBIND_2_5_AVAILABLE)
        if(NOT BUILD_SHARED_LIBS)
            set(install_tbbbind ON)
        endif()
    endif()
endif()

# install TBB

# define variables for OpenVINOConfig.cmake
if(THREADING MATCHES "^(TBB|TBB_AUTO)$")
    set(IE_TBB_DIR "${TBB_DIR}")
    list(APPEND PATH_VARS "IE_TBB_DIR")
endif()

if(install_tbbbind)
    set(IE_TBBBIND_DIR "${TBBBIND_2_5}")
    list(APPEND PATH_VARS "IE_TBBBIND_DIR")
endif()

# install only downloaded TBB, system one is not installed
if(THREADING MATCHES "^(TBB|TBB_AUTO)$" AND TBBROOT MATCHES ${TEMP})
    ie_cpack_add_component(tbb REQUIRED)
    ie_cpack_add_component(tbb_dev REQUIRED)
    list(APPEND core_components tbb)
    list(APPEND core_dev_components tbb_dev)

    install(DIRECTORY "${TBB}/lib"
            DESTINATION runtime/3rdparty/tbb
            COMPONENT tbb)
    # Windows only
    if(EXISTS "${TBB}/bin")
        install(DIRECTORY "${TBB}/bin"
                DESTINATION runtime/3rdparty/tbb
                COMPONENT tbb)
    endif()
    install(FILES "${TBB}/LICENSE"
            DESTINATION runtime/3rdparty/tbb
            COMPONENT tbb)

    set(IE_TBB_DIR_INSTALL "3rdparty/tbb/cmake")
    install(FILES "${TBB}/cmake/TBBConfig.cmake"
                  "${TBB}/cmake/TBBConfigVersion.cmake"
            DESTINATION runtime/${IE_TBB_DIR_INSTALL}
            COMPONENT tbb_dev)
    install(DIRECTORY "${TBB}/include"
            DESTINATION runtime/3rdparty/tbb
            COMPONENT tbb_dev)
endif()

# install tbbbind for static OpenVINO case
if(install_tbbbind)
    install(DIRECTORY "${TBBBIND_2_5}/lib"
            DESTINATION runtime/3rdparty/tbb_bind_2_5
            COMPONENT tbb)
    install(FILES "${TBBBIND_2_5}/LICENSE"
            DESTINATION runtime/3rdparty/tbb_bind_2_5
            COMPONENT tbb)

    set(IE_TBBBIND_DIR_INSTALL "3rdparty/tbb_bind_2_5/cmake")
    install(FILES "${TBBBIND_2_5}/cmake/TBBBIND_2_5Config.cmake"
            DESTINATION runtime/${IE_TBBBIND_DIR_INSTALL}
            COMPONENT tbb_dev)
endif()
