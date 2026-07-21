# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CPackComponent)

if(OV_GENERATOR_MULTI_CONFIG)
    set(OPENVINO_STATIC_PDB_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/$<CONFIG>/compile_pdbs)
else()
    set(OPENVINO_STATIC_PDB_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/compile_pdbs)
endif()

#
# ov_install_pdb(<target name>)
#
macro(ov_install_pdb target)
    if(WIN32)
        get_target_property(_lib_type ${target} TYPE)

        if(BUILD_SHARED_LIBS)
            # check that target type is either MODULE or SHARED
            if(NOT _lib_type MATCHES "^(MODULE_LIBRARY|SHARED_LIBRARY)$")
                message(FATAL_ERROR "OpenVINO PDB files should be installed only for SHARED or MODULE libraries, given target type is ${_lib_type}")
            endif()

            # installation of linker PDB files for shared libraries
            install(FILES $<TARGET_PDB_FILE:${target}>
                    DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT pdb
                    OPTIONAL
                    EXCLUDE_FROM_ALL)
        elseif(_lib_type STREQUAL "STATIC_LIBRARY")
            get_target_property(_compile_pdb_name ${target} OUTPUT_NAME)
            if(_compile_pdb_name MATCHES "NOTFOUND")
                set(_compile_pdb_name ${target})
            endif()

            set_target_properties(${target} PROPERTIES
                                  COMPILE_PDB_NAME ${_compile_pdb_name}
                                  COMPILE_PDB_NAME_DEBUG ${_compile_pdb_name}${OV_DEBUG_POSTFIX}
                                  COMPILE_PDB_OUTPUT_DIRECTORY "${OPENVINO_STATIC_PDB_OUTPUT_DIRECTORY}")

            # override compile PDB locations for objects libraries within static library
            get_target_property(sources ${target} SOURCES)
            foreach(source IN LISTS sources)
                if(source MATCHES "\\$<TARGET_OBJECTS:")
                    string(REGEX REPLACE ".*\\$<TARGET_OBJECTS:" "" source "${source}")
                    string(REGEX REPLACE ">$" "" source "${source}")
                    string(REGEX REPLACE ">$" "" object_library "${source}")

                    if(TARGET ${object_library})
                        # we need to rename CPU dnnl PDB files to be different from GPU ones (which we cannot rename on cmake level because of external project)
                        if(object_library MATCHES "^dnnl.*")
                            set(_compile_pdb_name "openvino_cpu_${object_library}")
                        else()
                            set(_compile_pdb_name ${object_library})
                        endif()

                        set_target_properties(${object_library} PROPERTIES
                                              COMPILE_PDB_NAME ${_compile_pdb_name}
                                              COMPILE_PDB_NAME_DEBUG ${_compile_pdb_name}${OV_DEBUG_POSTFIX}
                                              COMPILE_PDB_OUTPUT_DIRECTORY "${OPENVINO_STATIC_PDB_OUTPUT_DIRECTORY}")
                    endif()

                    unset(object_library)
                endif()
            endforeach()

            unset(source)
            unset(sources)
            unset(_compile_pdb_name)
        endif()

        unset(_lib_type)
    endif()
endmacro()

#
# _ov_collect_static_deps_impl(<target>)
#
# Internal recursive helper: walks both INTERFACE_LINK_LIBRARIES and
# LINK_LIBRARIES depth-first and accumulates all non-imported project targets
# (resolved through ALIAS) into the global property _OV_STATIC_DEP_RESULT.
# A companion global property _OV_STATIC_DEP_VISITED prevents re-visiting.
# Generator expressions in the dep lists are silently skipped.
#
function(_ov_collect_static_deps_impl target)
    # Resolve ALIAS target
    get_target_property(_csd_alias "${target}" ALIASED_TARGET)
    if(_csd_alias)
        set(_csd_real "${_csd_alias}")
    else()
        set(_csd_real "${target}")
    endif()

    if(NOT TARGET "${_csd_real}")
        return()
    endif()

    # Guard: skip already-visited targets
    get_property(_csd_visited GLOBAL PROPERTY _OV_STATIC_DEP_VISITED)
    if("${_csd_real}" IN_LIST _csd_visited)
        return()
    endif()
    set_property(GLOBAL APPEND PROPERTY _OV_STATIC_DEP_VISITED "${_csd_real}")

    # Skip IMPORTED targets — they belong to external packages (e.g. system TBB)
    get_target_property(_csd_imp "${_csd_real}" IMPORTED)
    if(_csd_imp)
        return()
    endif()

    set_property(GLOBAL APPEND PROPERTY _OV_STATIC_DEP_RESULT "${_csd_real}")

    # Collect both interface and private link deps.
    set(_csd_all_deps "")
    foreach(_csd_prop IN ITEMS INTERFACE_LINK_LIBRARIES LINK_LIBRARIES)
        get_target_property(_csd_prop_deps "${_csd_real}" "${_csd_prop}")
        if(_csd_prop_deps)
            list(APPEND _csd_all_deps ${_csd_prop_deps})
        endif()
    endforeach()

    foreach(_csd_dep IN LISTS _csd_all_deps)
        # Skip generator expressions such as $<LINK_ONLY:...>
        if(_csd_dep MATCHES "^\\$<")
            continue()
        endif()
        if(TARGET "${_csd_dep}")
            _ov_collect_static_deps_impl("${_csd_dep}")
        endif()
    endforeach()
endfunction()

#
# ov_install_static_deps(<targets-list-variable> <comp>)
#
# Installs every target named in <targets-list-variable> and all of their
# non-imported transitive link dependencies into the OpenVINO export set
# (OpenVINOTargets).
#
macro(ov_install_static_deps _ov_isd_targets_var _ov_isd_comp)
    set_property(GLOBAL PROPERTY _OV_STATIC_DEP_VISITED "")
    set_property(GLOBAL PROPERTY _OV_STATIC_DEP_RESULT "")
    foreach(_ov_isd_root IN LISTS ${_ov_isd_targets_var})
        if(TARGET "${_ov_isd_root}")
            _ov_collect_static_deps_impl("${_ov_isd_root}")
        endif()
    endforeach()
    get_property(_ov_isd_all GLOBAL PROPERTY _OV_STATIC_DEP_RESULT)
    foreach(_ov_isd_dep IN LISTS _ov_isd_all)
        ov_install_static_lib("${_ov_isd_dep}" "${_ov_isd_comp}")
    endforeach()
    unset(_ov_isd_targets_var)
    unset(_ov_isd_root)
    unset(_ov_isd_dep)
    unset(_ov_isd_all)
endmacro()

#
# ov_register_static_deps_in_export(<targets-list-variable> <export-set-name>)
#
# Registers every target in <targets-list-variable> and all of their
# non-imported transitive link dependencies into a NAMED export set without
# the BUILD_SHARED_LIBS guard that ov_install_static_deps has.
#
# Used to satisfy CMake's install(EXPORT ...) validation for third-party export
# sets (e.g. ONNXTargets) in shared builds, where ov_install_static_lib's
# if(NOT BUILD_SHARED_LIBS) guard would otherwise leave the deps unregistered.
#
# NOTE: Do NOT call this for OpenVINOTargets in static builds — use
# ov_install_static_deps instead, to avoid "exported multiple times" errors.
#
macro(ov_register_static_deps_in_export _ov_rsde_targets_var _ov_rsde_export)
    set_property(GLOBAL PROPERTY _OV_STATIC_DEP_VISITED "")
    set_property(GLOBAL PROPERTY _OV_STATIC_DEP_RESULT "")
    foreach(_ov_rsde_root IN LISTS ${_ov_rsde_targets_var})
        if(TARGET "${_ov_rsde_root}")
            _ov_collect_static_deps_impl("${_ov_rsde_root}")
        endif()
    endforeach()
    get_property(_ov_rsde_all GLOBAL PROPERTY _OV_STATIC_DEP_RESULT)
    foreach(_ov_rsde_dep IN LISTS _ov_rsde_all)
        get_target_property(_ov_rsde_alias "${_ov_rsde_dep}" ALIASED_TARGET)
        if(_ov_rsde_alias)
            set(_ov_rsde_real "${_ov_rsde_alias}")
        else()
            set(_ov_rsde_real "${_ov_rsde_dep}")
        endif()
        get_target_property(_ov_rsde_imp "${_ov_rsde_real}" IMPORTED)
        if(NOT _ov_rsde_imp)
            install(TARGETS "${_ov_rsde_real}"
                    EXPORT "${_ov_rsde_export}"
                    ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR}
                    COMPONENT ${OV_CPACK_COMP_CORE}
                    ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
        endif()
    endforeach()
    unset(_ov_rsde_root)
    unset(_ov_rsde_dep)
    unset(_ov_rsde_real)
    unset(_ov_rsde_alias)
    unset(_ov_rsde_imp)
    unset(_ov_rsde_all)
endmacro()

#
# ov_install_static_lib(<target> <comp>)
#
macro(ov_install_static_lib target comp)
    if(NOT BUILD_SHARED_LIBS)
        # Resolve alias targets — install(TARGETS) does not accept aliases
        get_target_property(aliased_target ${target} ALIASED_TARGET)
        if(aliased_target)
            set(install_target ${aliased_target})
        else()
            set(install_target ${target})
        endif()

        # Skip imported targets — they are owned by their own package config
        get_target_property(is_imported ${install_target} IMPORTED)
        if(NOT is_imported)
            get_target_property(target_type ${install_target} TYPE)
            if(target_type STREQUAL "STATIC_LIBRARY")
                set_target_properties(${install_target} PROPERTIES EXCLUDE_FROM_ALL OFF)
            endif()

            # save all internal installed targets to filter them later in 'ov_generate_dev_package_config'
            list(APPEND openvino_installed_targets ${install_target})
            set(openvino_installed_targets "${openvino_installed_targets}" CACHE INTERNAL
                "A list of OpenVINO internal targets" FORCE)

            install(TARGETS ${install_target} EXPORT OpenVINOTargets
                    ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${comp} ${ARGN})

            # install compile PDB file as well
            ov_install_pdb(${install_target})

            # export to local tree to build against static build tree
            export(TARGETS ${install_target} NAMESPACE openvino::
                   APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
        endif()
    endif()
endmacro()

#
# ov_set_install_rpath(<target> <lib_install_path> <dependency_install_path> ...)
#
# macOS:
# Sets LC_RPATH properties for macOS MACH-O binaries to ensure that libraries can find their dependencies
# when macOS system integrity protection (SIP) is enabled (DYLD_LIBRARY_PATH is ignored in this case).
# Note, that this is important when binaries are dynamically loaded at runtime (e.g. via Python).
#
# NPM:
# we need to set RPATH, because archive must be self-sufficient
#
function(ov_set_install_rpath TARGET_NAME lib_install_path)
    if(APPLE AND CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|TZST|ZIP)$" OR CPACK_GENERATOR STREQUAL "NPM")
        if (APPLE)
            set(RPATH_PREFIX "@loader_path")
        else()
            set(RPATH_PREFIX "$ORIGIN")
        endif()

        unset(rpath_list)
        foreach(dependency_install_path IN LISTS ARGN)
            file(RELATIVE_PATH dependency_rpath "/${lib_install_path}" "/${dependency_install_path}")
            set(dependency_rpath "${RPATH_PREFIX}/${dependency_rpath}")
            list(APPEND rpath_list "${dependency_rpath}")
        endforeach()

        set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${rpath_list}")
    endif()
endfunction()

#
# ov_get_pyversion(<OUT pyversion>)
#
function(ov_get_pyversion pyversion)
    find_package(Python3 QUIET COMPONENTS Interpreter Develoment.Module)
    if(Python3_Interpreter_FOUND)
        set(_pyversion "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
        if(Python3_SOABI AND (Python3_SOABI MATCHES "cpython-[0-9]+t-" OR Python3_SOABI MATCHES "cp[0-9]+t-"))
            set(_pyversion "${_pyversion}t")
        endif()
        set(${pyversion} "${_pyversion}" PARENT_SCOPE)
    else()
        set(${pyversion} "NOT-FOUND" PARENT_SCOPE)
    endif()
endfunction()

#
# ov_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal OV list
#
function(ov_cpack_add_component name)
    if(NOT ${name} IN_LIST OV_CPACK_COMPONENTS_ALL)
        cpack_add_component(${name} ${ARGN})

        # need to store informarion about cpack_add_component arguments in CMakeCache.txt
        # to restore it later
        set(_${name}_cpack_component_args "${ARGN}" CACHE INTERNAL "Argument for cpack_add_component for ${name} cpack component" FORCE)

        list(APPEND OV_CPACK_COMPONENTS_ALL ${name})
        set(OV_CPACK_COMPONENTS_ALL "${OV_CPACK_COMPONENTS_ALL}" CACHE INTERNAL "" FORCE)
    endif()
endfunction()

foreach(comp IN LISTS OV_CPACK_COMPONENTS_ALL)
    unset(_${comp}_cpack_component_args)
endforeach()
unset(OV_CPACK_COMPONENTS_ALL CACHE)

# create `tests` component
if(ENABLE_TESTS)
    cpack_add_component(tests HIDDEN)
endif()

#
#  ov_install_with_name(<FILE> <COMPONENT>)
#
# if <FILE> is a symlink, we resolve it, but install file with a name of symlink
#
function(ov_install_with_name file component)
    get_filename_component(actual_name "${file}" NAME)
    if((APPLE AND actual_name MATCHES "^[^\.]+\.[0-9]+${CMAKE_SHARED_LIBRARY_SUFFIX}$") OR
                (actual_name MATCHES "^.*\.${CMAKE_SHARED_LIBRARY_SUFFIX}\.[0-9]+$"))
        if(IS_SYMLINK "${file}")
            get_filename_component(file "${file}" REALPATH)
            set(install_rename RENAME "${actual_name}")
        endif()

        install(FILES "${file}"
                DESTINATION runtime/3rdparty/${component}/lib
                COMPONENT ${component}
                EXCLUDE_FROM_ALL
                ${install_rename})

        set("${component}_INSTALLED" ON PARENT_SCOPE)
    endif()
endfunction()

#
# checks that current OpenVINO versions has previous version in RPM / DEB conflicts
#
function(ov_check_conflicts_versions var_name)
    set(ov_major ${OpenVINO_VERSION_MAJOR})
    set(ov_minor ${OpenVINO_VERSION_MINOR})
    set(ov_patch ${OpenVINO_VERSION_PATCH})

    if(ov_patch EQUAL 0)
        if(ov_minor EQUAL 0)
            math(EXPR ov_major "${ov_major} - 1")
        else()
            math(EXPR ov_minor "${ov_minor} - 1")
        endif()
    else()
        math(EXPR ov_patch "${ov_patch} - 1")
    endif()

    set(ov_prev_version "${ov_major}.${ov_minor}.${ov_patch}")

    # perform check
    if(NOT ov_prev_version IN_LIST ${var_name})
        message(FATAL_ERROR "List ${var_name} (${${var_name}}) does not contain version ${ov_prev_version}")
    endif()
endfunction()

#
# List of public OpenVINO components
#

macro(ov_define_component_names)
    # core components
    set(OV_CPACK_COMP_CORE "core")
    set(OV_CPACK_COMP_CORE_C "core_c")
    set(OV_CPACK_COMP_CORE_DEV "core_dev")
    set(OV_CPACK_COMP_CORE_C_DEV "core_c_dev")
    # licensing
    set(OV_CPACK_COMP_LICENSING "licensing")
    # samples
    set(OV_CPACK_COMP_CPP_SAMPLES "cpp_samples")
    set(OV_CPACK_COMP_C_SAMPLES "c_samples")
    set(OV_CPACK_COMP_PYTHON_SAMPLES "python_samples")
    # python
    set(OV_CPACK_COMP_PYTHON_OPENVINO "pyopenvino")
    set(OV_CPACK_COMP_BENCHMARK_APP "benchmark_app")
    set(OV_CPACK_COMP_OVC "ovc")
    set(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE "pyopenvino_package")
    set(OV_CPACK_COMP_PYTHON_WHEELS "python_wheels")
    set(OV_CPACK_COMP_OPENVINO_REQ_FILES "openvino_req_files")
    # nodejs
    set(OV_CPACK_COMP_NPM "ov_node_addon")
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES "install_dependencies")
    set(OV_CPACK_COMP_SETUPVARS "setupvars")
    # symbolic_links
    set(OV_CPACK_COMP_LINKS "core_dev_links")
    # pkgconfig
    set(OV_CPACK_COMP_PKG_CONFIG "core_dev_pkgconfig")
endmacro()

ov_define_component_names()

#
# Include generator specific configuration file:
# 1. Overrides directories set by ov_<debian | rpm | archive | common_libraries>_cpack_set_dirs()
#    This is requried, because different generator use different locations for installed files
# 2. Merges some components using ov_override_component_names()
#    This is required, because different generators have different set of components
#    (e.g. C and C++ API are separate components)
# 3. Exclude some components using ov_define_component_include_rules()
#    This steps exclude some files from installation by defining variables meaning EXCLUDE_ALL
# 4. Sets ov_<debian | rpm | ...>_specific_settings() with DEB generator variables
#    This 'callback' is later called from ov_cpack (wrapper for standard cpack) to set
#    per-component settings (e.g. package names, dependencies, versions and system dependencies)
# 5. (Optional) Defines the following helper functions, which can be used by 3rdparty modules:
#    Debian:
#     - ov_debian_add_changelog_and_copyright()
#     - ov_debian_add_lintian_suppression()
#     - ov_debian_generate_conflicts()
#     - ov_debian_add_latest_component()
#    RPM:
#     - ov_rpm_add_rpmlint_suppression()
#     - ov_rpm_generate_conflicts()
#     - ov_rpm_copyright()
#     - ov_rpm_add_latest_component()
#
if(CPACK_GENERATOR STREQUAL "DEB")
    include(packaging/debian/debian)
elseif(CPACK_GENERATOR STREQUAL "RPM")
    include(packaging/rpm/rpm)
elseif(CPACK_GENERATOR STREQUAL "NSIS")
    include(packaging/nsis)
elseif(CPACK_GENERATOR STREQUAL "NPM")
    include(packaging/npm)
elseif(CPACK_GENERATOR MATCHES "^(CONDA-FORGE|BREW|CONAN|VCPKG)$")
    include(packaging/common-libraries)
elseif(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|TZST|ZIP)$")
    include(packaging/archive)
endif()

macro(ov_cpack)
    set(CPACK_SOURCE_GENERATOR "") # not used
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenVINO™ Toolkit")
    set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED OFF)
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")
    set(CPACK_PACKAGE_CONTACT "OpenVINO Developers <openvino@intel.com>")
    set(CPACK_VERBATIM_VARIABLES ON)
    set(CPACK_COMPONENTS_ALL ${ARGN})

    # default permissions for directories creation
    set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE OWNER_EXECUTE
        WORLD_READ WORLD_EXECUTE)

    # archive operations can be run in parallel since CMake 3.20
    set(CPACK_THREADS 8)

    if(NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()

    # TODO: replace with openvino and handle multi-config generators case
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
    endif()

    set(CPACK_PACKAGE_VERSION "${OpenVINO_VERSION}")
    # build version can be empty in case we are running cmake out of git repository
    if(NOT OpenVINO_VERSION_BUILD STREQUAL "000")
        set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.${OpenVINO_VERSION_BUILD}")
    endif()

    foreach(ver MAJOR MINOR PATCH)
        if(DEFINED OpenVINO_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${OpenVINO_VERSION_${ver}})
        else()
            message(FATAL_ERROR "Internal: OpenVINO_VERSION_${ver} variable is not defined")
        endif()
    endforeach()

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    # include GENERATOR dedicated per-component configuration file
    # NOTE: private modules need to define ov_cpack_settings macro
    # for custom packages configuration
    if(COMMAND ov_cpack_settings)
        ov_cpack_settings()
    endif()

    include(CPack)
endmacro()
