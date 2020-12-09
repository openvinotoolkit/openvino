# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(CI_BUILD_NUMBER      "2021.3.0-1923-bbbeb29612c-mryzhov/samples_reqs")

set(WHEEL_PACKAGE_NAME "openvino" CACHE STRING "Name of the package")
set(WHEEL_LICENCE_TYPE "Proprietary - Intel" CACHE STRING "License for the package")
set(WHEEL_AUTOR "Intel Corporation" CACHE STRING "Package author’s name")
set(WHEEL_AUTOR_EMAIL "openvino_pushbot@intel.com" CACHE STRING "Email address of the package author")
set(WHEEL_DESC "Inference Engine Python* API" CACHE STRING "Short, summary description of the package")
set(WHEEL_VERSION "0.0.0" CACHE STRING "Version of this release")
set(WHEEL_BUILD "000" CACHE STRING "Build number of this package")

string(REGEX MATCH "([0-9?]+)\\.([0-9?]+)\\.([0-9?]+)" WHEEL_VERSION "${CI_BUILD_NUMBER}")
string(REGEX MATCH "-([0-9?]+)-" WHEEL_BUILD "${CI_BUILD_NUMBER}")

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(LIBS_CPACK_PATH "libs")

# runtime libs
set(LIB_DIRS    ${CMAKE_INSTALL_PREFIX}/${PYTHON_BRIDGE_CPACK_PATH}/${LIBS_CPACK_PATH})
# python bindings
set(BINDINGS_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_BRIDGE_CPACK_PATH})

# prepare setup.py
configure_file(${SETUP_PY_IN} ${SETUP_PY})

if(UNIX)
    set(CMAKE_INSTALL_RPATH "../libs")
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "../libs")
endif()

#
set(runtime_targets inference_engine
                    inference_engine_legacy
                    inference_engine_transformations
                    inference_engine_preproc
                    inference_engine_lp_transformations
                    inference_engine_plugin_api
                    inference_engine_ir_reader
                    ${NGRAPH_LIBRARIES}
)

if(NGRAPH_ONNX_IMPORT_ENABLE)
    list(APPEND runtime_targets onnx_importer)
    list(APPEND runtime_targets inference_engine_onnx_reader)
endif()

set(target_files $<TARGET_FILE_DIR:inference_engine>/plugins.xml)

set(plugin_targets  MultiDevicePlugin
                    HeteroPlugin
)

if(ENABLE_MKL_DNN)
    list(APPEND plugin_targets MKLDNNPlugin)
endif()

if(ENABLE_CLDNN)
    list(APPEND plugin_targets clDNNPlugin)
    list(APPEND target_files $<TARGET_FILE_DIR:clDNNPlugin>/cache.json)
endif()

add_custom_target(ie_wheel ALL DEPENDS ${runtime_targets} ${plugin_targets})

# install
install(FILES ${SETUP_PY} ${CMAKE_CURRENT_SOURCE_DIR}/pypi_overview.md
    DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}
    COMPONENT ${PYTHON_VERSION}
)

install(TARGETS ${runtime_targets}
                ${plugin_targets}
        RUNTIME DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${LIBS_CPACK_PATH}
        COMPONENT ${PYTHON_VERSION}
)

install(FILES ${target_files}
        DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${LIBS_CPACK_PATH}
        COMPONENT ${PYTHON_VERSION})

if(EXISTS "${TBB}/bin")
    file(GLOB tbb_libs
         LIST_DIRECTORIES false
         "${TBB}/bin/*.so"
         "${TBB}/bin/*.dll"
         "${TBB}/bin/*.dylib"
    )
    install(FILES ${tbb_libs}
            DESTINATION ${PYTHON_BRIDGE_CPACK_PATH}/${LIBS_CPACK_PATH}
            COMPONENT ${PYTHON_VERSION}
    )
endif()
