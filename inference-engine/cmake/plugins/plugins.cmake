# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)

set(PLUGIN_FILES "" CACHE INTERNAL "")

function(get_shared_library_name target_name library_name)
    set(LIB_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
    set(LIB_SUFFIX "${IE_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}")

    set("${library_name}" "${LIB_PREFIX}${target_name}${LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

if(NOT TARGET ie_plugins)
    add_custom_target(ie_plugins)
endif()

#
# ie_add_plugin(NAME <targetName>
#               DEVICE_NAME <deviceName>
#               SOURCES <sources>
#               OBJECT_LIBRARIES <object_libs>
#               VERSION_DEFINES_FOR <source>
#               SKIP_INSTALL
#               )
#
function(ie_add_plugin)
    set(options SKIP_INSTALL)
    set(oneValueArgs NAME DEVICE_NAME VERSION_DEFINES_FOR)
    set(multiValueArgs SOURCES OBJECT_LIBRARIES CPPLINT_FILTERS)
    cmake_parse_arguments(IE_PLUGIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IE_PLUGIN_NAME)
        message(FATAL_ERROR "Please, specify plugin target name")
    endif()

    if(NOT IE_PLUGIN_DEVICE_NAME)
        message(FATAL_ERROR "Please, specify device name for ${IE_PLUGIN_NAME}")
    endif()

    # create and configure target

    if(IE_PLUGIN_VERSION_DEFINES_FOR)
        addVersionDefines(${IE_PLUGIN_VERSION_DEFINES_FOR} CI_BUILD_NUMBER)
    endif()

    set(input_files ${IE_PLUGIN_SOURCES})
    foreach(obj_lib IN LISTS IE_PLUGIN_OBJECT_LIBRARIES)
        list(APPEND input_files $<TARGET_OBJECTS:${obj_lib}>)
        add_cpplint_target(${obj_lib}_cpplint FOR_TARGETS ${obj_lib})
    endforeach()

    add_library(${IE_PLUGIN_NAME} SHARED ${input_files})
    target_compile_definitions(${IE_PLUGIN_NAME} PRIVATE IMPLEMENT_INFERENCE_ENGINE_PLUGIN)

    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "Inference Engine ${IE_PLUGIN_DEVICE_NAME} device plugin library")

    if(TARGET IE::inference_engine_plugin_api)
        target_link_libraries(${IE_PLUGIN_NAME} PRIVATE IE::inference_engine_plugin_api)
    else()
        target_link_libraries(${IE_PLUGIN_NAME} PRIVATE inference_engine_plugin_api)
    endif()

    if(WIN32)
        set_target_properties(${IE_PLUGIN_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
    endif()

    set(custom_filter "")
    foreach(filter IN LISTS IE_PLUGIN_CPPLINT_FILTERS)
        string(CONCAT custom_filter "${custom_filter}" "," "${filter}")
    endforeach()

    add_cpplint_target(${IE_PLUGIN_NAME}_cpplint FOR_TARGETS ${IE_PLUGIN_NAME} CUSTOM_FILTERS ${custom_filter})

    # append plugin to the list to register

    list(APPEND PLUGIN_FILES "${IE_PLUGIN_DEVICE_NAME}:${IE_PLUGIN_NAME}")
    list(REMOVE_DUPLICATES PLUGIN_FILES)
    set(PLUGIN_FILES "${PLUGIN_FILES}" CACHE INTERNAL "" FORCE)

    add_dependencies(ie_plugins ${IE_PLUGIN_NAME})
    if(TARGET inference_engine_preproc)
        add_dependencies(${IE_PLUGIN_NAME} inference_engine_preproc)
    endif()

    # fake dependencies to build in the following order:
    # IE -> IE readers -> IE inference plugins -> IE-based apps
    if(TARGET inference_engine_ir_reader)
        add_dependencies(${IE_PLUGIN_NAME} inference_engine_ir_reader)
    endif()
    if(TARGET inference_engine_ir_v7_reader)
        add_dependencies(${IE_PLUGIN_NAME} inference_engine_ir_v7_reader)
    endif()
    if(TARGET inference_engine_onnx_reader)
        add_dependencies(${IE_PLUGIN_NAME} inference_engine_onnx_reader)
    endif()

    # install rules

    if(NOT IE_PLUGIN_SKIP_INSTALL)
        string(TOLOWER "${IE_PLUGIN_DEVICE_NAME}" install_component)
        ie_cpack_add_component(${install_component} REQUIRED DEPENDS core)

        install(TARGETS ${IE_PLUGIN_NAME}
            RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ${install_component}
            ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ${install_component}
            LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ${install_component})
    endif()
endfunction()

#
# ie_register_plugins(MAIN_TARGET <main target name>
#                     POSSIBLE_PLUGINS <list of plugins which can be build by this repo>)
#
macro(ie_register_plugins)
    set(options)
    set(oneValueArgs MAIN_TARGET)
    set(multiValueArgs POSSIBLE_PLUGINS)
    cmake_parse_arguments(IE_REGISTER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT IE_REGISTER_MAIN_TARGET)
        message(FATAL_ERROR "Please, define MAIN_TARGET")
    endif()

    set(plugins_to_remove ${IE_REGISTER_POSSIBLE_PLUGINS})
    set(plugin_files_local)
    set(config_output_file "$<TARGET_FILE_DIR:${IE_REGISTER_MAIN_TARGET}>/plugins.xml")

    foreach(plugin IN LISTS plugins_to_remove)
        add_custom_command(TARGET ${IE_REGISTER_MAIN_TARGET} POST_BUILD
                  COMMAND
                    "${CMAKE_COMMAND}"
                    -D "IE_CONFIG_OUTPUT_FILE=${config_output_file}"
                    -D "IE_PLUGIN_NAME=${plugin}"
                    -D "IE_CONFIGS_DIR=${CMAKE_BINARY_DIR}/plugins"
                    -P "${IE_MAIN_SOURCE_DIR}/cmake/plugins/unregister_plugin_cmake.cmake"
                  COMMENT
                    "Remove ${plugin} from the plugins.xml file"
                  VERBATIM)
    endforeach()

    foreach(name IN LISTS PLUGIN_FILES)
        string(REPLACE ":" ";" name "${name}")
        list(LENGTH name length)
        if(NOT ${length} EQUAL 2)
            message(FATAL_ERROR "Unexpected error, please, contact developer of this script")
        endif()
        list(GET name 0 device_name)
        list(GET name 1 name)

        # create plugin file
        set(config_file_name "${CMAKE_BINARY_DIR}/plugins/${name}.xml")
        get_shared_library_name(${name} library_name)

        add_custom_command(TARGET ${IE_REGISTER_MAIN_TARGET} POST_BUILD
           COMMAND
              "${CMAKE_COMMAND}"
              -D "IE_CONFIG_OUTPUT_FILE=${config_file_name}"
              -D "IE_DEVICE_NAME=${device_name}"
              -D "IE_PLUGIN_LIBRARY_NAME=${library_name}"
              -P "${IE_MAIN_SOURCE_DIR}/cmake/plugins/create_plugin_file.cmake"
          COMMENT "Register ${name} plugin"
          VERBATIM)

        list(APPEND plugin_files_local "${config_file_name}")
    endforeach()

    add_custom_command(TARGET ${IE_REGISTER_MAIN_TARGET} POST_BUILD
                      COMMAND
                        "${CMAKE_COMMAND}"
                        -D "CMAKE_SHARED_LIBRARY_PREFIX=${CMAKE_SHARED_LIBRARY_PREFIX}"
                        -D "IE_CONFIG_OUTPUT_FILE=${config_output_file}"
                        -D "IE_CONFIGS_DIR=${CMAKE_BINARY_DIR}/plugins"
                        -P "${IE_MAIN_SOURCE_DIR}/cmake/plugins/register_plugin_cmake.cmake"
                      COMMENT
                        "Registering plugins to plugins.xml config file"
                      VERBATIM)
endmacro()
