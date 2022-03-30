# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

set(PLUGIN_FILES "" CACHE INTERNAL "")

function(ie_plugin_get_file_name target_name library_name)
    set(LIB_PREFIX "${CMAKE_SHARED_MODULE_PREFIX}")
    set(LIB_SUFFIX "${IE_BUILD_POSTFIX}${CMAKE_SHARED_MODULE_SUFFIX}")

    set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

if(NOT TARGET ie_plugins)
    add_custom_target(ie_plugins)
endif()

#
# ie_add_plugin(NAME <targetName>
#               DEVICE_NAME <deviceName>
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
function(ie_add_plugin)
    set(options SKIP_INSTALL ADD_CLANG_FORMAT AS_EXTENSION SKIP_REGISTRATION)
    set(oneValueArgs NAME DEVICE_NAME VERSION_DEFINES_FOR PSEUDO_PLUGIN_FOR)
    set(multiValueArgs DEFAULT_CONFIG SOURCES OBJECT_LIBRARIES CPPLINT_FILTERS)
    cmake_parse_arguments(IE_PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IE_PLUGIN_NAME)
        message(FATAL_ERROR "Please, specify plugin target name")
    endif()

    if(NOT IE_PLUGIN_DEVICE_NAME)
        message(FATAL_ERROR "Please, specify device name for ${IE_PLUGIN_NAME}")
    endif()

    # create and configure target

    if(NOT IE_PLUGIN_PSEUDO_PLUGIN_FOR)
        if(IE_PLUGIN_VERSION_DEFINES_FOR)
            addVersionDefines(${IE_PLUGIN_VERSION_DEFINES_FOR} CI_BUILD_NUMBER)
        endif()

        set(input_files ${IE_PLUGIN_SOURCES})
        foreach(obj_lib IN LISTS IE_PLUGIN_OBJECT_LIBRARIES)
            list(APPEND input_files $<TARGET_OBJECTS:${obj_lib}>)
            add_cpplint_target(${obj_lib}_cpplint FOR_TARGETS ${obj_lib})
        endforeach()

        if(BUILD_SHARED_LIBS)
            set(library_type MODULE)
        else()
            set(library_type STATIC)
        endif()

        add_library(${IE_PLUGIN_NAME} ${library_type} ${input_files})

        target_compile_definitions(${IE_PLUGIN_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)
        if(NOT BUILD_SHARED_LIBS)
            # to distinguish functions creating plugin objects
            target_compile_definitions(${IE_PLUGIN_NAME} PRIVATE
                IE_CREATE_PLUGIN=CreatePluginEngine${IE_PLUGIN_DEVICE_NAME})
            if(IE_PLUGIN_AS_EXTENSION)
                # to distinguish functions creating extensions objects
                target_compile_definitions(${IE_PLUGIN_NAME} PRIVATE
                    IE_CREATE_EXTENSION=CreateExtensionShared${IE_PLUGIN_DEVICE_NAME})
            endif()
        endif()

        ie_add_vs_version_file(NAME ${IE_PLUGIN_NAME}
            FILEDESCRIPTION "Inference Engine ${IE_PLUGIN_DEVICE_NAME} device plugin library")

        target_link_libraries(${IE_PLUGIN_NAME} PRIVATE openvino::runtime openvino::runtime::dev)

        if(WIN32)
            set_target_properties(${IE_PLUGIN_NAME} PROPERTIES COMPILE_PDB_NAME ${IE_PLUGIN_NAME})
        endif()

        if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_CROSSCOMPILING)
            target_link_options(${IE_PLUGIN_NAME} PRIVATE -Wl,--unresolved-symbols=ignore-in-shared-libs)
        endif()

        set(custom_filter "")
        foreach(filter IN LISTS IE_PLUGIN_CPPLINT_FILTERS)
            string(CONCAT custom_filter "${custom_filter}" "," "${filter}")
        endforeach()

        if (IE_PLUGIN_ADD_CLANG_FORMAT)
            add_clang_format_target(${IE_PLUGIN_NAME}_clang FOR_TARGETS ${IE_PLUGIN_NAME})
        else()
            add_cpplint_target(${IE_PLUGIN_NAME}_cpplint FOR_TARGETS ${IE_PLUGIN_NAME} CUSTOM_FILTERS ${custom_filter})
        endif()

        add_dependencies(ie_plugins ${IE_PLUGIN_NAME})
        if(TARGET openvino_gapi_preproc)
            if(BUILD_SHARED_LIBS)
                add_dependencies(${IE_PLUGIN_NAME} openvino_gapi_preproc)
            else()
                target_link_libraries(${IE_PLUGIN_NAME} PRIVATE openvino_gapi_preproc)
            endif()
        endif()

        # fake dependencies to build in the following order:
        # IE -> IE readers -> IE inference plugins -> IE-based apps
        if(BUILD_SHARED_LIBS)
            if(TARGET openvino_ir_frontend)
                add_dependencies(${IE_PLUGIN_NAME} openvino_ir_frontend)
            endif()
            if(TARGET openvino_onnx_frontend)
                add_dependencies(${IE_PLUGIN_NAME} openvino_onnx_frontend)
            endif()
            if(TARGET openvino_paddle_frontend)
                add_dependencies(${IE_PLUGIN_NAME} openvino_paddle_frontend)
            endif()
            if(TARGET openvino_tensorflow_frontend)
                add_dependencies(${IE_PLUGIN_NAME} openvino_tensorflow_frontend)
            endif()
            # TODO: remove with legacy CNNNLayer API / IR v7
            if(TARGET inference_engine_ir_v7_reader)
                add_dependencies(${IE_PLUGIN_NAME} inference_engine_ir_v7_reader)
            endif()
        endif()

        # install rules
        if(NOT IE_PLUGIN_SKIP_INSTALL OR NOT BUILD_SHARED_LIBS)
            string(TOLOWER "${IE_PLUGIN_DEVICE_NAME}" install_component)
            ie_cpack_add_component(${install_component} REQUIRED DEPENDS core)

            if(BUILD_SHARED_LIBS)
                install(TARGETS ${IE_PLUGIN_NAME}
                        LIBRARY DESTINATION ${IE_CPACK_RUNTIME_PATH}
                        COMPONENT ${install_component})
            else()
                ov_install_static_lib(${IE_PLUGIN_NAME} ${install_component})
            endif()
        endif()
    endif()

    # Enable for static build to generate correct plugins.hpp
    if(NOT IE_PLUGIN_SKIP_REGISTRATION OR NOT BUILD_SHARED_LIBS)
        # check that plugin with such name is not registered
        foreach(plugin_entry IN LISTS PLUGIN_FILES)
            string(REPLACE ":" ";" plugin_entry "${plugin_entry}")
            list(GET plugin_entry -1 library_name)
            list(GET plugin_entry 0 plugin_name)
            if(plugin_name STREQUAL "${IE_PLUGIN_DEVICE_NAME}" AND
                    NOT library_name STREQUAL ${IE_PLUGIN_NAME})
                message(FATAL_ERROR "${IE_PLUGIN_NAME} and ${library_name} are both registered as ${plugin_name}")
            endif()
        endforeach()

        # append plugin to the list to register

        list(APPEND PLUGIN_FILES "${IE_PLUGIN_DEVICE_NAME}:${IE_PLUGIN_NAME}")
        set(PLUGIN_FILES "${PLUGIN_FILES}" CACHE INTERNAL "" FORCE)
        set(${IE_PLUGIN_DEVICE_NAME}_CONFIG "${IE_PLUGIN_DEFAULT_CONFIG}" CACHE INTERNAL "" FORCE)
        set(${IE_PLUGIN_DEVICE_NAME}_PSEUDO_PLUGIN_FOR "${IE_PLUGIN_PSEUDO_PLUGIN_FOR}" CACHE INTERNAL "" FORCE)
        set(${IE_PLUGIN_DEVICE_NAME}_AS_EXTENSION "${IE_PLUGIN_AS_EXTENSION}" CACHE INTERNAL "" FORCE)
    endif()
endfunction()

function(ov_add_plugin)
    ie_add_plugin(${ARGN})
endfunction()

#
# ie_register_plugins_dynamic(MAIN_TARGET <main target name>)
#
macro(ie_register_plugins_dynamic)
    set(options)
    set(oneValueArgs MAIN_TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(IE_REGISTER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IE_REGISTER_MAIN_TARGET)
        message(FATAL_ERROR "Please, define MAIN_TARGET")
    endif()

    # Unregister <device_name>.xml files for plugins from current build tree

    set(config_output_file "$<TARGET_FILE_DIR:${IE_REGISTER_MAIN_TARGET}>/plugins.xml")

    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()
        list(GET name 0 device_name)
        add_custom_command(TARGET ${IE_REGISTER_MAIN_TARGET} POST_BUILD
                  COMMAND
                    "${CMAKE_COMMAND}"
                    -D "IE_CONFIG_OUTPUT_FILE=${config_output_file}"
                    -D "IE_PLUGIN_NAME=${device_name}"
                    -D "IE_CONFIGS_DIR=${CMAKE_BINARY_DIR}/plugins"
                    -P "${IEDevScripts_DIR}/plugins/unregister_plugin_cmake.cmake"
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
        ie_plugin_get_file_name(${name} library_name)

        add_custom_command(TARGET ${IE_REGISTER_MAIN_TARGET} POST_BUILD
           COMMAND
              "${CMAKE_COMMAND}"
              -D "IE_CONFIG_OUTPUT_FILE=${config_file_name}"
              -D "IE_DEVICE_NAME=${device_name}"
              -D "IE_PLUGIN_PROPERTIES=${${device_name}_CONFIG}"
              -D "IE_PLUGIN_LIBRARY_NAME=${library_name}"
              -P "${IEDevScripts_DIR}/plugins/create_plugin_file.cmake"
          COMMENT "Register ${device_name} device as ${library_name}"
          VERBATIM)

        list(APPEND plugin_files_local "${config_file_name}")
    endforeach()

    # Combine all <device_name>.xml files into plugins.xml

    add_custom_command(TARGET ${IE_REGISTER_MAIN_TARGET} POST_BUILD
                      COMMAND
                        "${CMAKE_COMMAND}"
                        -D "CMAKE_SHARED_MODULE_PREFIX=${CMAKE_SHARED_MODULE_PREFIX}"
                        -D "IE_CONFIG_OUTPUT_FILE=${config_output_file}"
                        -D "IE_CONFIGS_DIR=${CMAKE_BINARY_DIR}/plugins"
                        -P "${IEDevScripts_DIR}/plugins/register_plugin_cmake.cmake"
                      COMMENT
                        "Registering plugins to plugins.xml config file"
                      VERBATIM)
endmacro()

#
# ie_register_plugins()
#
macro(ie_register_plugins)
    if(BUILD_SHARED_LIBS)
        ie_register_plugins_dynamic(${ARGN})
    endif()
endmacro()

#
# ov_register_plugins()
#
macro(ov_register_plugins)
    if(BUILD_SHARED_LIBS)
        ie_register_plugins_dynamic(${ARGN})
    endif()
endmacro()

#
# ie_target_link_plugins(<TARGET_NAME>)
#
function(ie_target_link_plugins TARGET_NAME)
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
# ie_generate_plugins_hpp()
#
function(ie_generate_plugins_hpp)
    if(BUILD_SHARED_LIBS)
        return()
    endif()

    set(device_mapping)
    set(device_configs)
    set(as_extension)
    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()

        # create device mapping: preudo device => actual device
        list(GET name 0 device_name)
        if(${device_name}_PSEUDO_PLUGIN_FOR)
            list(APPEND device_mapping "${device_name}:${${device_name}_PSEUDO_PLUGIN_FOR}")
        else()
            list(APPEND device_mapping "${device_name}:${device_name}")
        endif()

        # register plugin as extension
        if(${device_name}_AS_EXTENSION)
            list(APPEND as_extension -D "${device_name}_AS_EXTENSION=ON")
        endif()

        # add default plugin config options
        if(${device_name}_CONFIG)
            list(APPEND device_configs -D "${device_name}_CONFIG=${${device_name}_CONFIG}")
        endif()
    endforeach()

    # add plugins to libraries including ie_plugins.hpp
    ie_target_link_plugins(openvino)
    if(TARGET inference_engine_s)
        ie_target_link_plugins(inference_engine_s)
    endif()

    set(ie_plugins_hpp "${CMAKE_BINARY_DIR}/src/inference/ie_plugins.hpp")
    set(plugins_hpp_in "${IEDevScripts_DIR}/plugins/plugins.hpp.in")

    add_custom_command(OUTPUT "${ie_plugins_hpp}"
                       COMMAND
                        "${CMAKE_COMMAND}"
                        -D "IE_DEVICE_MAPPING=${device_mapping}"
                        -D "IE_PLUGINS_HPP_HEADER_IN=${plugins_hpp_in}"
                        -D "IE_PLUGINS_HPP_HEADER=${ie_plugins_hpp}"
                        ${device_configs}
                        ${as_extension}
                        -P "${IEDevScripts_DIR}/plugins/create_plugins_hpp.cmake"
                       DEPENDS
                         "${plugins_hpp_in}"
                         "${IEDevScripts_DIR}/plugins/create_plugins_hpp.cmake"
                       COMMENT
                         "Generate ie_plugins.hpp for static build"
                       VERBATIM)

    # for some reason dependency on source files does not work
    # so, we have to use explicit target and make it dependency for inference_engine
    add_custom_target(_ie_plugins_hpp DEPENDS ${ie_plugins_hpp})
    add_dependencies(inference_engine_obj _ie_plugins_hpp)

    # add dependency for object files
    get_target_property(sources inference_engine_obj SOURCES)
    foreach(source IN LISTS sources)
        if("${source}" MATCHES "\\$\\<TARGET_OBJECTS\\:([A-Za-z0-9_]*)\\>")
            # object library
            set(obj_library ${CMAKE_MATCH_1})
            get_target_property(obj_sources ${obj_library} SOURCES)
            list(APPEND all_sources ${obj_sources})
        else()
            # usual source
            list(APPEND all_sources ${source})
        endif()
    endforeach()

    # add dependency on header file generation for all inference_engine source files
    set_source_files_properties(${all_sources} PROPERTIES OBJECT_DEPENDS ${ie_plugins_hpp})
endfunction()
