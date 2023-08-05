# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(OpenVINOPython DESCRIPTION "OpenVINO Runtime Python bindings")

#
# Packages & settings
#

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(OpenVINODeveloperPackage REQUIRED
                 PATHS "${InferenceEngineDeveloperPackage_DIR}")
    set(OpenVINO_BINARY_DIR "${OpenVINODeveloperPackage_DIR}")
endif()

#
# Check python requirements
#

set(ov_python_req "${OpenVINOPython_SOURCE_DIR}/requirements.txt")
set(ie_python_req "${OpenVINOPython_SOURCE_DIR}/src/compatibility/openvino/requirements-dev.txt")

function(ov_check_python_build_conditions)
    # user explicitly specified ENABLE_PYTHON=ON
    if(ENABLE_PYTHON)
        set(find_package_mode REQUIRED)
        set(message_mode FATAL_ERROR)
    else()
        set(find_package_mode QUIET)
        set(message_mode WARNING)
    endif()

    # Try to find python3 and its libs
    find_host_package(PythonInterp 3 ${find_package_mode})
    if(PYTHONINTERP_FOUND)
        if(PYTHON_VERSION_MINOR GREATER_EQUAL 11)
            set(pybind11_min_version 2.9.2)
        else()
            set(pybind11_min_version 2.8.0)
        endif()
        if(CMAKE_CROSSCOMPILING)
            # since this version pybind11 has PYBIND11_PYTHONLIBS_OVERWRITE option
            # to properly cross-compile, users need to set:
            #  -DPYTHON_EXECUTABLE=<python interpreter of our choice version 3.X>
            #  -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libc-2.31.so
            #  -DPYTHON_INCLUDE_DIR=<path to includes for python 3.X>
            #  -DPYBIND11_PYTHONLIBS_OVERWRITE=OFF
            #  -DPYTHON_MODULE_EXTENSION=$(aarch64-linux-gnu-python3-config --extension-suffix)
            set(pybind11_min_version 2.10.1)
        endif()

        function(_ov_find_python_libs_new)
            set(pybind11_tools_dir "${OpenVINOPython_SOURCE_DIR}/thirdparty/pybind11/tools")
            if(EXISTS ${pybind11_tools_dir})
                list(APPEND CMAKE_MODULE_PATH ${pybind11_tools_dir})
            else()
                find_host_package(pybind11 ${pybind11_min_version} QUIET)
                if(pybind11_FOUND)
                    list(APPEND CMAKE_MODULE_PATH "${pybind11_DIR}")
                endif()
            endif()
            # use libraries with the same version as python itself
            set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION_STRING})
            find_host_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} EXACT ${find_package_mode})
            set(PYTHONLIBSNEW_FOUND ${PYTHONLIBS_FOUND} PARENT_SCOPE)
        endfunction()
        # try to find python libraries
        _ov_find_python_libs_new()
        if(PYTHONLIBSNEW_FOUND)
            # clear Python_ADDITIONAL_VERSIONS to find only python library matching PYTHON_EXECUTABLE
            unset(Python_ADDITIONAL_VERSIONS CACHE)
            find_host_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT ${find_package_mode})
        endif()
        if(NOT PYTHONLIBS_FOUND)
            message(${message_mode} "Python development libraries are not found. OpenVINO Python API will be turned off (ENABLE_PYTHON is OFF)")
        endif()
    else()
        message(${message_mode} "Python 3.x interpreter is not found. OpenVINO Python API will be turned off (ENABLE_PYTHON is OFF)")
    endif()

    # check pyopenvino requirements to OV 2.0 API
    ov_check_pip_packages(REQUIREMENTS_FILE ${ov_python_req}
                          RESULT_VAR ov_python_req_FOUND
                          WARNING_MESSAGE "install python3 -m pip install -r ${ov_python_req} for OV API 2.0 requirements"
                          MESSAGE_MODE TRACE)
    # ov_python_req are not mandatory for build
    set(ov_python_req_FOUND ON)

    # check for Cython requirement for build IE API 1.0
    ov_check_pip_packages(REQUIREMENTS_FILE ${ie_python_req}
                          RESULT_VAR ie_python_req_FOUND
                          WARNING_MESSAGE "install python3 -m pip install -r ${ie_python_req} for IE API 1.0 requirements"
                          MESSAGE_MODE TRACE)

    # cython can be installed as a debian package, so pip requirements can be unsatisfied
    # so, let's check to find cython anyway
    if(NOT ie_python_req_FOUND)
        find_package(Cython QUIET
                     PATHS "${OpenVINOPython_SOURCE_DIR}/src/compatibility/openvino/cmake"
                     NO_CMAKE_FIND_ROOT_PATH
                     NO_DEFAULT_PATH)
        if(CYTHON_VERSION VERSION_GREATER_EQUAL 0.29)
            set(ie_python_req_FOUND ON)
        else()
            message(${message_mode} "Python requirements '${ie_python_req}' are missed, IE Python API 1.0 will not be built (ENABLE_PYTHON is OFF)")
        endif()
    endif()

    if(NOT OV_GENERATOR_MULTI_CONFIG AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_DEBUG_POSTFIX)
        set(python_debug ON)
        message(${message_mode} "Building python bindings in debug configuration is not supported on your platform (ENABLE_PYTHON is OFF)")
    else()
        set(python_debug OFF)
    endif()

    if(PYTHONLIBS_FOUND AND ov_python_req_FOUND AND ie_python_req_FOUND AND NOT python_debug)
        set(ENABLE_PYTHON_DEFAULT ON PARENT_SCOPE)
    else()
        set(ENABLE_PYTHON_DEFAULT OFF PARENT_SCOPE)
    endif()

    # to disable API 1.0
    set(ie_python_req_FOUND ${ie_python_req_FOUND} PARENT_SCOPE)
    # set pybind11 minimal version
    set(pybind11_min_version ${pybind11_min_version} PARENT_SCOPE)
endfunction()

ov_check_python_build_conditions()

ie_option(ENABLE_PYTHON "Enables OpenVINO Python API build" ${ENABLE_PYTHON_DEFAULT})

#
# Check for wheel package
#

# user explicitly specified ENABLE_WHEEL=ON
if(ENABLE_WHEEL)
    set(find_package_mode REQUIRED)
    set(message_mode FATAL_ERROR)
else()
    set(find_package_mode QUIET)
    set(message_mode WARNING)
endif()

set(wheel_reqs "${OpenVINOPython_SOURCE_DIR}/wheel/requirements-dev.txt")
ov_check_pip_packages(REQUIREMENTS_FILE "${OpenVINOPython_SOURCE_DIR}/wheel/requirements-dev.txt"
                      RESULT_VAR ENABLE_WHEEL_DEFAULT
                      MESSAGE_MODE WARNING)

if(LINUX)
    find_host_program(patchelf_program
                      NAMES patchelf
                      DOC "Path to patchelf tool")
    if(NOT patchelf_program)
        set(ENABLE_WHEEL_DEFAULT OFF)
        message(${message_mode} "patchelf is not found. It is required to build OpenVINO Runtime wheel. Install via apt-get install patchelf")
    endif()
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    set(SETUP_PY_REQUIREMENTS_FOUND ON)
else()
    message(${message_mode} "Cmake version 3.15 and higher is required to build 'openvino' wheel. Provided version ${CMAKE_VERSION}")
    set(SETUP_PY_REQUIREMENTS_FOUND OFF)
endif()

if(NOT SETUP_PY_REQUIREMENTS_FOUND)
    # setup.py requirements are importnant to build wheel
    set(ENABLE_WHEEL_DEFAULT OFF)
endif()

# this option should not be a part of OpenVINODeveloperPackage
# since wheels can be built only together with main OV build
ie_dependent_option(ENABLE_WHEEL "Build wheel packages for PyPI" ${ENABLE_WHEEL_DEFAULT} "ENABLE_PYTHON" OFF)

if(NOT ENABLE_PYTHON)
    if(CMAKE_SOURCE_DIR STREQUAL OpenVINOPython_SOURCE_DIR)
        message(FATAL_ERROR "Python OpenVINO API requirements are not satisfied. Please, install ${ie_python_req} and ${ov_python_req}")
    else()
        return()
    endif()
endif()

if(NOT SETUP_PY_REQUIREMENTS_FOUND AND ENABLE_PYTHON_PACKAGING)
    message(FATAL_ERROR "Python cannot be packaged, because setup.py requirements are not satisfied (cmake version >= 3.15 is required, provided ${CMAKE_VERSION})")
endif()

#
# Build the code
#

# TODO: find a real root cause and remove this workaround
# or localize it to the specific place of the code
add_definitions(-DPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF)

find_package(pybind11 ${pybind11_min_version} QUIET)

if(NOT pybind11_FOUND)
    add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)
endif()

add_subdirectory(src/compatibility/pyngraph)
add_subdirectory(src/pyopenvino)

if(ie_python_req_FOUND)
    add_subdirectory(src/compatibility/openvino)
else()
    message(WARNING "NOTE: Python API for OpenVINO 1.0 is disabled")
endif()

#
# Packaging
#

macro(ov_define_setup_py_packaging_vars)
    # PYTHON_VERSION_MAJOR and PYTHON_VERSION_MINOR are defined inside pybind11
    set(pyversion python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})

    # define version (syncronize with tools/openvino_dev/CMakeLists.txt)
    if(DEFINED ENV{CI_BUILD_DEV_TAG} AND NOT "$ENV{CI_BUILD_DEV_TAG}" STREQUAL "")
        set(WHEEL_VERSION "${OpenVINO_VERSION}.$ENV{CI_BUILD_DEV_TAG}" CACHE STRING "Version of this release" FORCE)
        set(wheel_pre_release ON)
    else()
        set(WHEEL_VERSION ${OpenVINO_VERSION} CACHE STRING "Version of this release" FORCE)
    endif()
    set(WHEEL_BUILD "${OpenVINO_VERSION_BUILD}" CACHE STRING "Build number of this release" FORCE)

    # Common vars used by setup.py
    set(PY_PACKAGES_DIR ${OV_CPACK_PYTHONDIR})
    set(TBB_LIBS_DIR runtime/3rdparty/tbb/lib)
    if(WIN32)
        set(TBB_LIBS_DIR runtime/3rdparty/tbb/bin)
    endif()
    set(PUGIXML_LIBS_DIR runtime/3rdparty/pugixml/lib)

    # define setup.py running environment
    set(setup_py_env ${CMAKE_COMMAND} -E env
        # for cross-compilation
        SETUPTOOLS_EXT_SUFFIX=${PYTHON_MODULE_EXTENSION}
        # versions
        WHEEL_VERSION=${WHEEL_VERSION}
        WHEEL_BUILD=${WHEEL_BUILD}
        # build locations
        OPENVINO_BINARY_DIR=${OpenVINO_BINARY_DIR}
        OPENVINO_PYTHON_BINARY_DIR=${OpenVINOPython_BINARY_DIR}
        # to create proper directories for BA, OVC tools
        CPACK_GENERATOR=${CPACK_GENERATOR}
        # variables to reflect cpack locations
        OV_RUNTIME_LIBS_DIR=${OV_WHEEL_RUNTIMEDIR}
        TBB_LIBS_DIR=${TBB_LIBS_DIR}
        PUGIXML_LIBS_DIR=${PUGIXML_LIBS_DIR}
        PY_PACKAGES_DIR=${PY_PACKAGES_DIR})
endmacro()

macro(ov_define_setup_py_dependencies)
    foreach(_target
            # Python API dependencies
            _pyngraph pyopenvino py_ov_frontends
            # legacy Python API 1.0 dependencies (remove before 2024.0 release)
            ie_api constants
            # plugins
            ov_plugins
            # frontends
            ov_frontends)
        if(TARGET ${_target})
            list(APPEND ov_setup_py_deps ${_target})
        endif()
    endforeach()

    file(GLOB_RECURSE compat_ngraph_py_files ${OpenVINOPython_SOURCE_DIR}/src/compatibility/*.py)
    file(GLOB_RECURSE openvino_py_files ${OpenVINOPython_SOURCE_DIR}/src/openvino/*.py)

    list(APPEND ov_setup_py_deps
        ${openvino_py_files}
        ${compat_ngraph_py_files}
        "${CMAKE_CURRENT_SOURCE_DIR}/wheel/setup.py"
        "${OpenVINOPython_SOURCE_DIR}/requirements.txt"
        "${OpenVINOPython_SOURCE_DIR}/wheel/readme.txt"
        "${OpenVINO_SOURCE_DIR}/LICENSE"
        "${OpenVINO_SOURCE_DIR}/licensing/onednn_third-party-programs.txt"
        "${OpenVINO_SOURCE_DIR}/licensing/runtime-third-party-programs.txt"
        "${OpenVINO_SOURCE_DIR}/licensing/tbb_third-party-programs.txt"
        "${OpenVINO_SOURCE_DIR}/docs/install_guides/pypi-openvino-rt.md")

    if(wheel_pre_release)
        list(APPEND ov_setup_py_deps
            "${OpenVINO_SOURCE_DIR}/docs/install_guides/pre-release-note.md")
    endif()
endmacro()

ov_define_setup_py_packaging_vars()
ov_define_setup_py_dependencies()

if(ENABLE_WHEEL)
    add_subdirectory(wheel)
endif()

#
# Target, which creates python install tree for cases of DEB | RPM packages
#

if(ENABLE_PYTHON_PACKAGING)
    # site-packages depending on package type
    set(python_xy "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    if(CPACK_GENERATOR STREQUAL "DEB")
        set(python_versioned_folder "python${PYTHON_VERSION_MAJOR}")
        set(ov_site_packages "dist-packages")
    elseif(CPACK_GENERATOR STREQUAL "RPM")
        set(python_versioned_folder "python${python_xy}")
        set(ov_site_packages "site-packages")
    endif()

    set(python_package_prefix "${CMAKE_CURRENT_BINARY_DIR}/install_${pyversion}")
    set(install_lib "${python_package_prefix}/lib/${python_versioned_folder}/${ov_site_packages}")
    set(meta_info_subdir "openvino-${OpenVINO_VERSION}-py${python_xy}.egg-info")
    set(meta_info_file "${install_lib}/${meta_info_subdir}/PKG-INFO")

    add_custom_command(OUTPUT ${meta_info_file}
        COMMAND ${CMAKE_COMMAND} -E remove_directory
            "${python_package_prefix}"
        COMMAND ${setup_py_env}
                # variables to reflect options (extensions only or full wheel package)
                PYTHON_EXTENSIONS_ONLY=ON
                SKIP_RPATH=ON
            "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/wheel/setup.py"
                --no-user-cfg
                --quiet
                build
                    --executable "/usr/bin/python3"
                build_ext
                install
                    --no-compile
                    --prefix "${python_package_prefix}"
                    --install-lib "${install_lib}"
                    --install-scripts "${python_package_prefix}/bin"
                    # --single-version-externally-managed
                    # --record=installed.txt
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS ${ov_setup_py_deps}
        COMMENT "Create python package with ${meta_info_subdir} folder")

    add_custom_target(_python_api_package ALL DEPENDS ${meta_info_file})

    # install python package, which will be later packed into DEB | RPM
    ov_cpack_add_component(${OV_CPACK_COMP_PYTHON_OPENVINO}_package_${pyversion} HIDDEN)

    install(DIRECTORY ${python_package_prefix}/
            DESTINATION ${CMAKE_INSTALL_PREFIX}
            COMPONENT ${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_${pyversion}
            ${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL}
            USE_SOURCE_PERMISSIONS)
endif()

#
# Tests
#

if(ENABLE_TESTS)
    add_subdirectory(tests/mock/mock_py_frontend)
    add_subdirectory(tests/mock/pyngraph_fe_mock_api)
endif()

if(OpenVINODeveloperPackage_FOUND)
    ie_cpack(${IE_CPACK_COMPONENTS_ALL})
endif()
