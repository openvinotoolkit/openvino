# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

set(PLUGIN_FILES "" CACHE INTERNAL "")

function(ov_plugin_get_file_name target_name library_name)
    set(LIB_PREFIX "${CMAKE_SHARED_MODULE_PREFIX}")
    set(LIB_SUFFIX "${OV_BUILD_POSTFIX}${CMAKE_SHARED_MODULE_SUFFIX}")

    get_target_property(LIB_NAME ${target_name} OUTPUT_NAME)
    if (LIB_NAME STREQUAL "LIB_NAME-NOTFOUND")
        set(LIB_NAME ${target_name})
    endif()
    set("${library_name}" "${LIB_PREFIX}${LIB_NAME}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

if(NOT TARGET ov_plugins)
    add_custom_target(ov_plugins)
endif()

#
# ov_add_plugin(NAME <targetName>
#               DEVICE_NAME <deviceName>
#               [PSEUDO_DEVICE]
#               [PSEUDO_PLUGIN_FOR <actual_device>]
#               [AS_EXTENSION]
#               [DEFAULT_CONFIG <key:value;...>]
#               [SOURCES <sources>]
#               [OBJECT_LIBRARIES <object_libs>]
#               [VERSION_DEFINES_FOR <source>]
#               [SKIP_INSTALL]
#               [SKIP_REGISTRATION] Skip creation of <device>.xml
#               [ADD_CLANG_FORMAT]
#               )
#
function(ov_add_plugin)
    set(options SKIP_INSTALL PSEUDO_DEVICE ADD_CLANG_FORMAT AS_EXTENSION SKIP_REGISTRATION)
    set(oneValueArgs NAME DEVICE_NAME VERSION_DEFINES_FOR PSEUDO_PLUGIN_FOR)
    set(multiValueArgs DEFAULT_CONFIG SOURCES OBJECT_LIBRARIES CPPLINT_FILTERS)
    cmake_parse_arguments(OV_PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT OV_PLUGIN_NAME)
        message(FATAL_ERROR "Please, specify plugin target name")
    endif()

    if(NOT OV_PLUGIN_DEVICE_NAME)
        message(FATAL_ERROR "Please, specify device name for ${OV_PLUGIN_NAME}")
    endif()

    # create and configure target

    if(NOT OV_PLUGIN_PSEUDO_PLUGIN_FOR)
        set(input_files ${OV_PLUGIN_SOURCES})
        foreach(obj_lib IN LISTS OV_PLUGIN_OBJECT_LIBRARIES)
            list(APPEND input_files $<TARGET_OBJECTS:${obj_lib}>)
            add_cpplint_target(${obj_lib}_cpplint FOR_TARGETS ${obj_lib})
        endforeach()

        if(BUILD_SHARED_LIBS)
            set(library_type MODULE)
        else()
            set(library_type STATIC)
        endif()

        add_library(${OV_PLUGIN_NAME} ${library_type} ${input_files})

        if(OV_PLUGIN_VERSION_DEFINES_FOR)
            ov_add_version_defines(${OV_PLUGIN_VERSION_DEFINES_FOR} ${OV_PLUGIN_NAME})
        endif()

        target_compile_definitions(${OV_PLUGIN_NAME} PRIVATE IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
        if(NOT BUILD_SHARED_LIBS)
            # to distinguish functions creating plugin objects
            target_compile_definitions(${OV_PLUGIN_NAME} PRIVATE
                OV_CREATE_PLUGIN=create_plugin_engine_${OV_PLUGIN_DEVICE_NAME})
            if(OV_PLUGIN_AS_EXTENSION)
                # to distinguish functions creating extensions objects
                target_compile_definitions(${OV_PLUGIN_NAME} PRIVATE
                    OV_CREATE_EXTENSION=create_extensions_${OV_PLUGIN_DEVICE_NAME})
            endif()
        endif()

        ov_add_vs_version_file(NAME ${OV_PLUGIN_NAME}
            FILEDESCRIPTION "OpenVINO Runtime ${OV_PLUGIN_DEVICE_NAME} device plugin library")

        target_link_libraries(${OV_PLUGIN_NAME} PRIVATE openvino::runtime openvino::runtime::dev)

        if(WIN32)
            set_target_properties(${OV_PLUGIN_NAME} PROPERTIES COMPILE_PDB_NAME ${OV_PLUGIN_NAME})
        endif()

        if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_CROSSCOMPILING)
            if (APPLE)
                target_link_options(${OV_PLUGIN_NAME} PRIVATE -Wl,-undefined,dynamic_lookup)
            else()
                target_link_options(${OV_PLUGIN_NAME} PRIVATE -Wl,--unresolved-symbols=ignore-in-shared-libs)
            endif()
        endif()

        set(custom_filter "")
        foreach(filter IN LISTS OV_PLUGIN_CPPLINT_FILTERS)
            string(CONCAT custom_filter "${custom_filter}" "," "${filter}")
        endforeach()

        if (OV_PLUGIN_ADD_CLANG_FORMAT)
            ov_add_clang_format_target(${OV_PLUGIN_NAME}_clang FOR_SOURCES ${OV_PLUGIN_SOURCES})
        else()
            add_cpplint_target(${OV_PLUGIN_NAME}_cpplint FOR_TARGETS ${OV_PLUGIN_NAME} CUSTOM_FILTERS ${custom_filter})
        endif()

        # plugins does not have to be CXX ABI free, because nobody links with plugins,
        # but let's add this mark to see how it goes
        ov_abi_free_target(${OV_PLUGIN_NAME})

        add_dependencies(ov_plugins ${OV_PLUGIN_NAME})

        # install rules
        if(NOT OV_PLUGIN_SKIP_INSTALL OR NOT BUILD_SHARED_LIBS)
            string(TOLOWER "${OV_PLUGIN_DEVICE_NAME}" install_component)
            if(NOT BUILD_SHARED_LIBS)
                # in case of static libs everything is installed to 'core'
                set(install_component ${OV_CPACK_COMP_CORE})
            endif()

            if(OV_PLUGIN_PSEUDO_DEVICE)
                set(plugin_hidden HIDDEN)
            endif()
            ov_cpack_add_component(${install_component}
                                   DISPLAY_NAME "${OV_PLUGIN_DEVICE_NAME} runtime"
                                   DESCRIPTION "${OV_PLUGIN_DEVICE_NAME} runtime"
                                   ${plugin_hidden}
                                   DEPENDS ${OV_CPACK_COMP_CORE})

            if(BUILD_SHARED_LIBS)
                install(TARGETS ${OV_PLUGIN_NAME}
                        LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR}
                        COMPONENT ${install_component})
            else()
                ov_install_static_lib(${OV_PLUGIN_NAME} ${OV_CPACK_COMP_CORE})
            endif()
        endif()
    endif()

    # Enable for static build to generate correct plugins.hpp
    if(NOT OV_PLUGIN_SKIP_REGISTRATION OR NOT BUILD_SHARED_LIBS)
        # check that plugin with such name is not registered
        foreach(plugin_entry IN LISTS PLUGIN_FILES)
            string(REPLACE ":" ";" plugin_entry "${plugin_entry}")
            list(GET plugin_entry -1 library_name)
            list(GET plugin_entry 0 plugin_name)
            if(plugin_name STREQUAL "${OV_PLUGIN_DEVICE_NAME}" AND
                    NOT library_name STREQUAL ${OV_PLUGIN_NAME})
                message(FATAL_ERROR "${OV_PLUGIN_NAME} and ${library_name} are both registered as ${plugin_name}")
            endif()
        endforeach()

        # append plugin to the list to register

        list(APPEND PLUGIN_FILES "${OV_PLUGIN_DEVICE_NAME}:${OV_PLUGIN_NAME}")
        set(PLUGIN_FILES "${PLUGIN_FILES}" CACHE INTERNAL "" FORCE)
        set(${OV_PLUGIN_DEVICE_NAME}_CONFIG "${OV_PLUGIN_DEFAULT_CONFIG}" CACHE INTERNAL "" FORCE)
        set(${OV_PLUGIN_DEVICE_NAME}_PSEUDO_PLUGIN_FOR "${OV_PLUGIN_PSEUDO_PLUGIN_FOR}" CACHE INTERNAL "" FORCE)
        set(${OV_PLUGIN_DEVICE_NAME}_AS_EXTENSION "${OV_PLUGIN_AS_EXTENSION}" CACHE INTERNAL "" FORCE)
    endif()
endfunction()

#
# ov_register_in_plugins_xml(MAIN_TARGET <main target name>)
#
# Registers plugins in plugins.xml files for dynamic plugins build
#
macro(ov_register_in_plugins_xml)
    set(options)
    set(oneValueArgs MAIN_TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(OV_REGISTER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT OV_REGISTER_MAIN_TARGET)
        message(FATAL_ERROR "Please, define MAIN_TARGET")
    endif()

    # Unregister <device_name>.xml files for plugins from current build tree

    set(config_output_file "$<TARGET_FILE_DIR:${OV_REGISTER_MAIN_TARGET}>/plugins.xml")

    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()
        list(GET name 0 device_name)
        add_custom_command(TARGET ${OV_REGISTER_MAIN_TARGET} POST_BUILD
                  COMMAND
                    "${CMAKE_COMMAND}"
                    -D "OV_CONFIG_OUTPUT_FILE=${config_output_file}"
                    -D "OV_PLUGIN_NAME=${device_name}"
                    -D "OV_CONFIGS_DIR=${CMAKE_BINARY_DIR}/plugins"
                    -P "${OpenVINODeveloperScripts_DIR}/plugins/unregister_plugin_cmake.cmake"
                  COMMENT
                    "Remove ${device_name} from the plugins.xml file"
                  VERBATIM)
    endforeach()

    # Generate <device_name>.xml files

    set(plugin_files_local)
    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()
        list(GET name 0 device_name)
        list(GET name 1 name)

        # create plugin file
        set(config_file_name "${CMAKE_BINARY_DIR}/plugins/${device_name}.xml")
        ov_plugin_get_file_name(${name} library_name)

        add_custom_command(TARGET ${OV_REGISTER_MAIN_TARGET} POST_BUILD
           COMMAND
              "${CMAKE_COMMAND}"
              -D "OV_CONFIG_OUTPUT_FILE=${config_file_name}"
              -D "OV_DEVICE_NAME=${device_name}"
              -D "OV_PLUGIN_PROPERTIES=${${device_name}_CONFIG}"
              -D "OV_PLUGIN_LIBRARY_NAME=${library_name}"
              -P "${OpenVINODeveloperScripts_DIR}/plugins/create_plugin_file.cmake"
          COMMENT "Register ${device_name} device as ${library_name}"
          VERBATIM)

        list(APPEND plugin_files_local "${config_file_name}")
    endforeach()

    # Combine all <device_name>.xml files into plugins.xml

    add_custom_command(TARGET ${OV_REGISTER_MAIN_TARGET} POST_BUILD
                       COMMAND
                          "${CMAKE_COMMAND}"
                          -D "CMAKE_SHARED_MODULE_PREFIX=${CMAKE_SHARED_MODULE_PREFIX}"
                          -D "OV_CONFIG_OUTPUT_FILE=${config_output_file}"
                          -D "OV_CONFIGS_DIR=${CMAKE_BINARY_DIR}/plugins"
                          -P "${OpenVINODeveloperScripts_DIR}/plugins/register_plugin_cmake.cmake"
                        COMMENT
                          "Registering plugins to plugins.xml config file"
                        VERBATIM)
endmacro()

#
# ov_register_plugins()
#
macro(ov_register_plugins)
    if(BUILD_SHARED_LIBS AND ENABLE_PLUGINS_XML)
        ov_register_in_plugins_xml(${ARGN})
    endif()
endmacro()

#
# ov_target_link_plugins(<TARGET_NAME>)
#
function(ov_target_link_plugins TARGET_NAME)
    if(BUILD_SHARED_LIBS)
        return()
    endif()

    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()

        # link plugin to ${TARGET_NAME} static version
        list(GET name 1 plugin_name)
        target_link_libraries(${TARGET_NAME} PRIVATE ${plugin_name})
    endforeach()
endfunction()

#
# ov_generate_plugins_hpp()
#
# Generates plugins.hpp file for:
# - static plugins build
# - cases when plugins.xml file is disabled
#
function(ov_generate_plugins_hpp)
    set(device_mapping)
    set(device_configs)
    set(as_extension)
    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()

        # create device mapping: pseudo device => actual device
        list(GET name 0 device_name)
        if(BUILD_SHARED_LIBS)
            list(GET name 1 library_name)
            ov_plugin_get_file_name(${library_name} library_name)
            list(APPEND device_mapping "${device_name}:${library_name}")
        else()
            if(${device_name}_PSEUDO_PLUGIN_FOR)
                list(APPEND device_mapping "${device_name}:${${device_name}_PSEUDO_PLUGIN_FOR}")
            else()
                list(APPEND device_mapping "${device_name}:${device_name}")
            endif()

            # register plugin as extension
            if(${device_name}_AS_EXTENSION)
                list(APPEND as_extension -D "${device_name}_AS_EXTENSION=ON")
            endif()
        endif()

        # add default plugin config options
        if(${device_name}_CONFIG)
            # Replace ; to @ in order to have list inside list
            string(REPLACE ";" "@" config "${${device_name}_CONFIG}")
            list(APPEND device_configs -D "${device_name}_CONFIG=${config}")
        endif()
    endforeach()

    # add plugins to libraries including ov_plugins.hpp
    ov_target_link_plugins(openvino)
    if(TARGET openvino_runtime_s)
        ov_target_link_plugins(openvino_runtime_s)
    endif()

    get_target_property(OV_RUNTIME_OBJ_BINARY_DIR openvino_runtime_obj BINARY_DIR)
    if(OV_GENERATOR_MULTI_CONFIG AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
        set(ov_plugins_hpp "${OV_RUNTIME_OBJ_BINARY_DIR}/$<CONFIG>/ov_plugins.hpp")
    else()
        set(ov_plugins_hpp "${OV_RUNTIME_OBJ_BINARY_DIR}/ov_plugins.hpp")
    endif()
    set(plugins_hpp_in "${OpenVINODeveloperScripts_DIR}/plugins/plugins.hpp.in")

    add_custom_command(OUTPUT "${ov_plugins_hpp}"
                       COMMAND
                        "${CMAKE_COMMAND}"
                        -D "BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
                        -D "OV_DEVICE_MAPPING=${device_mapping}"
                        -D "OV_PLUGINS_HPP_HEADER_IN=${plugins_hpp_in}"
                        -D "OV_PLUGINS_HPP_HEADER=${ov_plugins_hpp}"
                        ${device_configs}
                        ${as_extension}
                        -P "${OpenVINODeveloperScripts_DIR}/plugins/create_plugins_hpp.cmake"
                       DEPENDS
                         "${plugins_hpp_in}"
                         "${OpenVINODeveloperScripts_DIR}/plugins/create_plugins_hpp.cmake"
                       COMMENT
                         "Generate ov_plugins.hpp"
                       VERBATIM)

    # for some reason dependency on source files does not work
    # so, we have to use explicit target and make it dependency for openvino_runtime_obj
    add_custom_target(_ov_plugins_hpp DEPENDS ${ov_plugins_hpp})
    add_dependencies(openvino_runtime_obj _ov_plugins_hpp)
endfunction()
