# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(WIN32)
    # 3.16: FindPython3.cmake can find Python via -DPython3_EXECUTABLE
    # 3.18: FindPython3.cmake can find Python automatically from virtualenv
    cmake_minimum_required(VERSION 3.16)
else()
    # 3.13: default choice
    cmake_minimum_required(VERSION 3.13)
endif()

project(OpenVINOPython DESCRIPTION "OpenVINO Runtime Python bindings")

#
# Packages & settings
#

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(OpenVINODeveloperPackage REQUIRED
                 PATHS "${InferenceEngineDeveloperPackage_DIR}")

    # we assume that OpenVINODeveloperPackage is generated in OpenVINO build tree
    set(OpenVINO_BINARY_DIR "${OpenVINODeveloperPackage_DIR}")
    # but this can be invalid for cases of OpenVINODeveloperPackage relocatable installation
    # so, we need to disable wheen generation for this case
    if(NOT EXISTS "${OpenVINO_BINARY_DIR}/cmake_install.cmake")
        set(OpenVINODeveloperPackage_RELOCATABLE ON)
    endif()

    set(OpenVINO_SOURCE_DIR "${OpenVINOPython_SOURCE_DIR}/../../../")
endif()

if(NOT DEFINED OpenVINODeveloperPackage_RELOCATABLE)
    set(OpenVINODeveloperPackage_RELOCATABLE OFF)
endif()

#
# Settings for FindPython3.cmake
#

if(NOT DEFINED Python3_USE_STATIC_LIBS)
    set(Python3_USE_STATIC_LIBS OFF)
endif()

if(NOT DEFINED Python3_FIND_VIRTUALENV)
    set(Python3_FIND_VIRTUALENV FIRST)
endif()

if(NOT DEFINED Python3_FIND_IMPLEMENTATIONS)
    set(Python3_FIND_IMPLEMENTATIONS CPython PyPy)
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(python3_development_component Development.Module)
else()
    set(python3_development_component Development)
endif()

#
# Check python requirements
#

set(ie_build_python_req "${OpenVINOPython_SOURCE_DIR}/src/compatibility/openvino/requirements-dev.txt")

function(ov_check_python_build_conditions)
    # user explicitly specified ENABLE_PYTHON=ON
    if(ENABLE_PYTHON)
        set(find_package_mode REQUIRED)
        set(message_mode FATAL_ERROR)
    else()
        set(find_package_mode QUIET)
        set(message_mode WARNING)
    endif()

    find_package(Python3 ${find_package_mode} COMPONENTS Interpreter ${python3_development_component})
    if(Python3_Development.Module_FOUND OR Python3_Development_FOUND)
        message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
        message(STATUS "Python3 version: ${Python3_VERSION}")
        if(Python3_PyPy_VERSION)
            message(STATUS "Python3 PyPy version: ${Python3_PyPy_VERSION}")
        endif()
        message(STATUS "Python3 interpreter ID: ${Python3_INTERPRETER_ID}")
        if(Python3_SOABI)
            message(STATUS "Python3 SOABI: ${Python3_SOABI}")
        endif()
        if(PYTHON_MODULE_EXTENSION)
            message(STATUS "PYTHON_MODULE_EXTENSION: ${PYTHON_MODULE_EXTENSION}")
        endif()
        message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
        message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
    else()
        message(${message_mode} "Python 3.x Interpreter and Development.Module components are not found. OpenVINO Python API will be turned off (ENABLE_PYTHON is OFF)")
    endif()

    # check for Cython requirement for build IE API 1.0
    ov_check_pip_packages(REQUIREMENTS_FILE ${ie_build_python_req}
                          RESULT_VAR ie_build_python_req_FOUND
                          WARNING_MESSAGE "install python3 -m pip install -r ${ie_build_python_req} for IE API 1.0 requirements"
                          MESSAGE_MODE TRACE)

    # cython can be installed as a debian package, so pip requirements can be unsatisfied
    # so, let's check to find cython anyway
    if(NOT ie_build_python_req_FOUND)
        find_package(Cython QUIET
                     PATHS "${OpenVINOPython_SOURCE_DIR}/src/compatibility/openvino/cmake"
                     NO_CMAKE_FIND_ROOT_PATH
                     NO_DEFAULT_PATH)
        if(CYTHON_VERSION VERSION_GREATER_EQUAL 0.29)
            set(ie_build_python_req_FOUND ON)
        else()
            message(${message_mode} "Python requirements '${ie_build_python_req}' are missed, IE Python API 1.0 will not be built (ENABLE_PYTHON is OFF)")
        endif()
    endif()

    if(NOT OV_GENERATOR_MULTI_CONFIG AND CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_DEBUG_POSTFIX)
        set(python_debug ON)
        message(${message_mode} "Building python bindings in debug configuration is not supported on your platform (ENABLE_PYTHON is OFF)")
    else()
        set(python_debug OFF)
    endif()

    if((Python3_Development.Module_FOUND OR Python3_Development_FOUND) AND
        ie_build_python_req_FOUND AND NOT python_debug)
        set(ENABLE_PYTHON_DEFAULT ON PARENT_SCOPE)
    else()
        set(ENABLE_PYTHON_DEFAULT OFF PARENT_SCOPE)
    endif()

    # to disable API 1.0
    set(ie_build_python_req_FOUND ${ie_build_python_req_FOUND} PARENT_SCOPE)
endfunction()

ov_check_python_build_conditions()

# check __init__.py files alignment

function(ov_check_init_files_alignment init_files)
    # check the files in pairs
    list(LENGTH init_files init_files_count)
    math(EXPR file_loop_range "${init_files_count}-2")
    foreach(init_file_idx RANGE 0 ${file_loop_range})
        math(EXPR init_file_idx_next "${init_file_idx}+1")
        list(GET init_files ${init_file_idx} file1)
        list(GET init_files ${init_file_idx_next} file2)

        execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files ${file1} ${file2}
            RESULT_VARIABLE compare_result
        )
        if(compare_result EQUAL 1)
            message(FATAL_ERROR "The runtime __init__.py files are misaligned: ${file1} and ${file2}")
        endif()
    endforeach()
endfunction()

set(INIT_FILES_RUNTIME "${OpenVINOPython_SOURCE_DIR}/src/openvino/__init__.py"
                       "${OpenVINOPython_SOURCE_DIR}/src/compatibility/openvino/__init__.py"
                       "${OpenVINO_SOURCE_DIR}/tools/ovc/openvino/__init__.py"
                       "${OpenVINO_SOURCE_DIR}/tools/benchmark_tool/openvino/__init__.py")

ov_check_init_files_alignment("${INIT_FILES_RUNTIME}")

ov_option(ENABLE_PYTHON "Enables OpenVINO Python API build" ${ENABLE_PYTHON_DEFAULT})

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
ov_dependent_option(ENABLE_WHEEL "Build wheel packages for PyPI" ${ENABLE_WHEEL_DEFAULT} "ENABLE_PYTHON;NOT OpenVINODeveloperPackage_RELOCATABLE" OFF)

if(NOT ENABLE_PYTHON)
    if(CMAKE_SOURCE_DIR STREQUAL OpenVINOPython_SOURCE_DIR)
        message(FATAL_ERROR "Python OpenVINO API build requirements are not satisfied. Please, install ${ie_build_python_req}")
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

if(Python3_VERSION_MINOR GREATER_EQUAL 11)
    set(pybind11_min_version 2.9.2)
else()
    set(pybind11_min_version 2.8.0)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter ${python3_development_component})
find_package(pybind11 ${pybind11_min_version} QUIET)

if(NOT pybind11_FOUND)
    add_subdirectory(thirdparty/pybind11 EXCLUDE_FROM_ALL)
endif()

add_subdirectory(src/compatibility/pyngraph)
add_subdirectory(src/pyopenvino)

if(ie_build_python_req_FOUND)
    add_subdirectory(src/compatibility/openvino)
else()
    message(WARNING "NOTE: Python API for OpenVINO 1.0 is disabled")
endif()

#
# Packaging
#

macro(ov_define_setup_py_packaging_vars)
    # Python3_VERSION_MAJOR and Python3_VERSION_MINOR are defined inside pybind11
    set(pyversion python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

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

    if(USE_BUILD_TYPE_SUBFOLDER)
        set(build_type ${CMAKE_BUILD_TYPE})
    else()
        set(build_type $<CONFIG>)
    endif()

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
        # propogate build type
        BUILD_TYPE=${build_type}
        # variables to reflect cpack locations
        OV_RUNTIME_LIBS_DIR=${OV_CPACK_RUNTIMEDIR}
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
    set(python_xy "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    if(CPACK_GENERATOR STREQUAL "DEB")
        set(python_versioned_folder "python${Python3_VERSION_MAJOR}")
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
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/wheel/setup.py"
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
                    --single-version-externally-managed
                    --record=installed.txt
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
    # provides a callback function to describe each component in repo
    include("${OpenVINO_SOURCE_DIR}/cmake/packaging/packaging.cmake")

    ov_cpack(${OV_CPACK_COMPONENTS_ALL})
endif()
