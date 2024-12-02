# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(FRONTEND_INSTALL_INCLUDE "${OV_CPACK_INCLUDEDIR}")
set(FRONTEND_NAME_PREFIX "openvino_")
set(FRONTEND_NAME_SUFFIX "_frontend")

set(FRONTEND_NAMES "" CACHE INTERNAL "")

if(NOT TARGET ov_frontends)
    add_custom_target(ov_frontends)
endif()

#
# ov_target_link_frontends(<TARGET_NAME>)
#
function(ov_target_link_frontends TARGET_NAME)
    if(BUILD_SHARED_LIBS)
        return()
    endif()

    foreach(name IN LISTS FRONTEND_NAMES)
        set(frontend_target_name "${FRONTEND_NAME_PREFIX}${name}${FRONTEND_NAME_SUFFIX}")
        target_link_libraries(${TARGET_NAME} PRIVATE ${frontend_target_name})
    endforeach()
endfunction()

#
# ov_generate_frontends_hpp()
#
function(ov_generate_frontends_hpp)
    if(BUILD_SHARED_LIBS)
        return()
    endif()

    # add frontends to libraries including ov_frontends.hpp
    ov_target_link_frontends(openvino)

    set(ov_frontends_hpp "${CMAKE_BINARY_DIR}/src/frontends/common/src/ov_frontends.hpp")
    set(frontends_hpp_in "${OpenVINODeveloperScripts_DIR}/frontends/ov_frontends.hpp.in")

    add_custom_command(OUTPUT "${ov_frontends_hpp}"
                       COMMAND
                        "${CMAKE_COMMAND}"
                        -D "OV_FRONTENDS_HPP_HEADER_IN=${frontends_hpp_in}"
                        -D "OV_FRONTENDS_HPP_HEADER=${ov_frontends_hpp}"
                        -D "FRONTEND_NAMES=${FRONTEND_NAMES}"
                        -P "${OpenVINODeveloperScripts_DIR}/frontends/create_frontends_hpp.cmake"
                       DEPENDS
                         "${frontends_hpp_in}"
                         "${OpenVINODeveloperScripts_DIR}/frontends/create_frontends_hpp.cmake"
                       COMMENT
                         "Generate ov_frontends.hpp for static build"
                       VERBATIM)

    # for some reason dependency on source files does not work
    # so, we have to use explicit target and make it dependency for frontend_common
    add_custom_target(_ov_frontends_hpp DEPENDS ${ov_frontends_hpp})
    add_dependencies(openvino_frontend_common_obj _ov_frontends_hpp)

    # add dependency for object files
    get_target_property(sources openvino_frontend_common_obj SOURCES)
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

    # add dependency on header file generation for all openvino_frontend_common_obj source files
    set_source_files_properties(${all_sources} PROPERTIES OBJECT_DEPENDS ${ov_frontends_hpp})
endfunction()

unset(protobuf_lite_installed CACHE)
unset(protobuf_installed CACHE)

#
# ov_frontend_group_files(<ROOT_DIR>  # Root path for scanning
#                         <REL_PATH>  # Relative path (in ROOT_DIR) is used for scanning
#                         <FILE_EXT>) # File extension for grouping
#
macro(ov_frontend_group_files root_dir rel_path file_mask)
   file(GLOB items RELATIVE ${root_dir}/${rel_path} ${root_dir}/${rel_path}/*)
   foreach(item ${items})
        if(IS_DIRECTORY ${root_dir}/${rel_path}/${item})
            ov_frontend_group_files(${root_dir} ${rel_path}/${item} ${file_mask})
        else()
            if(${item} MATCHES ".*\.${file_mask}$")
                string(REPLACE "/" "\\" groupname ${rel_path})
                source_group(${groupname} FILES ${root_dir}/${rel_path}/${item})
            endif()
        endif()
   endforeach()
endmacro()

#
# ov_add_frontend(NAME <IR|ONNX|...>
#                 FILEDESCRIPTION <description> # used on Windows to describe DLL file
#                 [LINKABLE_FRONTEND] # whether we can use FE API directly or via FEM only
#                 [SKIP_INSTALL] # private frontend, not for end users
#                 [DISABLE_CPP_INSTALL] # excludes frontend from all cmake rules
#                 [PROTOBUF_REQUIRED] # options to denote that protobuf is used
#                 [PROTOBUF_LITE] # requires only libprotobuf-lite
#                 [SKIP_NCC_STYLE] # use custom NCC rules
#                 [LINK_LIBRARIES <lib1 lib2 ...>])
#
macro(ov_add_frontend)
    set(options LINKABLE_FRONTEND PROTOBUF_REQUIRED PROTOBUF_LITE SKIP_NCC_STYLE SKIP_INSTALL DISABLE_CPP_INSTALL)
    set(oneValueArgs NAME FILEDESCRIPTION)
    set(multiValueArgs LINK_LIBRARIES PROTO_FILES)
    cmake_parse_arguments(OV_FRONTEND "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(prop NAME FILEDESCRIPTION)
        if(NOT DEFINED OV_FRONTEND_${prop})
            message(FATAL_ERROR "Frontend ${prop} property is not defined")
        endif()
    endforeach()

    set(TARGET_NAME "${FRONTEND_NAME_PREFIX}${OV_FRONTEND_NAME}${FRONTEND_NAME_SUFFIX}")

    list(APPEND FRONTEND_NAMES ${OV_FRONTEND_NAME})
    set(FRONTEND_NAMES "${FRONTEND_NAMES}" CACHE INTERNAL "" FORCE)

    set(frontend_root_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    if(frontend_root_dir MATCHES ".*src$")
        get_filename_component(frontend_root_dir "${frontend_root_dir}" DIRECTORY)
    endif()

    file(GLOB_RECURSE LIBRARY_SRC ${frontend_root_dir}/src/*.cpp)
    file(GLOB_RECURSE LIBRARY_HEADERS ${frontend_root_dir}/src/*.hpp)
    file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${frontend_root_dir}/include/*.hpp)

    set(${TARGET_NAME}_INCLUDE_DIR ${frontend_root_dir}/include)

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj

    ov_frontend_group_files(${frontend_root_dir}/ "src" "cpp")
    ov_frontend_group_files(${frontend_root_dir}/ "src" "proto")
    source_group("include" FILES ${LIBRARY_HEADERS})
    source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

    # Generate protobuf file on build time for each '.proto' file in src/proto
    set(protofiles_root_dir "${frontend_root_dir}/src/proto")
    file(GLOB_RECURSE proto_files ${protofiles_root_dir}/*.proto)

    foreach(proto_file IN LISTS proto_files)
        # filter out standaard google proto files
        if(proto_file MATCHES ".*google.*")
            continue()
        endif()

        file(RELATIVE_PATH proto_file_relative "${CMAKE_SOURCE_DIR}" "${proto_file}")
        get_filename_component(FILE_WE ${proto_file} NAME_WE)
        file(RELATIVE_PATH relative_path ${protofiles_root_dir} ${proto_file})
        get_filename_component(relative_path ${relative_path} DIRECTORY)
        set(OUTPUT_PB_SRC ${CMAKE_CURRENT_BINARY_DIR}/${relative_path}/${FILE_WE}.pb.cc)
        set(OUTPUT_PB_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${relative_path}/${FILE_WE}.pb.h)
        add_custom_command(
                OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
                COMMAND ${PROTOC_EXECUTABLE} ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${protofiles_root_dir} ${proto_file}
                DEPENDS ${PROTOC_DEPENDENCY} ${proto_file}
                COMMENT "Running C++ protocol buffer compiler (${PROTOC_EXECUTABLE}) on ${proto_file_relative}"
                VERBATIM
                COMMAND_EXPAND_LISTS)
        list(APPEND PROTO_SRCS "${OUTPUT_PB_SRC}")
        list(APPEND PROTO_HDRS "${OUTPUT_PB_HEADER}")
    endforeach()

    file(GLOB flatbuffers_schema_files ${frontend_root_dir}/src/schema/*.fbs)
    foreach(flatbuffers_schema_file IN LISTS flatbuffers_schema_files)
        file(RELATIVE_PATH flatbuffers_schema_file_relative "${CMAKE_SOURCE_DIR}" "${flatbuffers_schema_file}")
        get_filename_component(FILE_WE "${flatbuffers_schema_file}" NAME_WE)
        set(OUTPUT_FC_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${FILE_WE}_generated.h)
        add_custom_command(
                OUTPUT "${OUTPUT_FC_HEADER}"
                COMMAND ${flatbuffers_COMPILER} ARGS -c --gen-mutable -o ${CMAKE_CURRENT_BINARY_DIR} ${flatbuffers_schema_file}
                DEPENDS ${flatbuffers_DEPENDENCY} ${flatbuffers_schema_file}
                COMMENT "Running C++ flatbuffers compiler (${flatbuffers_COMPILER}) on ${flatbuffers_schema_file_relative}"
                VERBATIM
                COMMAND_EXPAND_LISTS)
        list(APPEND PROTO_HDRS "${OUTPUT_FC_HEADER}")
    endforeach()

    # Disable all warnings for generated code
    set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES COMPILE_OPTIONS -w GENERATED ON)

    # Create library
    add_library(${TARGET_NAME} ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS}
                               ${PROTO_SRCS} ${PROTO_HDRS} ${flatbuffers_schema_files} ${proto_files})

    if(OV_FRONTEND_LINKABLE_FRONTEND)
        # create beautiful alias
        add_library(openvino::frontend::${OV_FRONTEND_NAME} ALIAS ${TARGET_NAME})
    endif()

    # Shutdown protobuf when unloading the frontend dynamic library
    if(OV_FRONTEND_PROTOBUF_REQUIRED AND BUILD_SHARED_LIBS)
        target_link_libraries(${TARGET_NAME} PRIVATE openvino::protobuf_shutdown)
    endif()

    if(NOT BUILD_SHARED_LIBS)
        # override default function names
        target_compile_definitions(${TARGET_NAME} PRIVATE
            "-Dget_front_end_data=get_front_end_data_${OV_FRONTEND_NAME}"
            "-Dget_api_version=get_api_version_${OV_FRONTEND_NAME}")
    endif()

    # remove -Wmissing-declarations warning, because of frontends implementation specific
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
        target_compile_options(${TARGET_NAME} PRIVATE -Wno-missing-declarations)
    endif()

    target_include_directories(${TARGET_NAME}
            PUBLIC
                $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
            PRIVATE
                $<TARGET_PROPERTY:openvino::frontend::common,INTERFACE_INCLUDE_DIRECTORIES>
                ${frontend_root_dir}/src
                ${CMAKE_CURRENT_BINARY_DIR})

    ov_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION ${OV_FRONTEND_FILEDESCRIPTION})

    target_link_libraries(${TARGET_NAME} PRIVATE ${OV_FRONTEND_LINK_LIBRARIES} PUBLIC openvino::runtime)
    ov_add_library_version(${TARGET_NAME})

    if(OV_FRONTEND_PROTOBUF_REQUIRED)
        # WA for TF frontends which always require protobuf (not protobuf-lite)
        # if TF FE is built in static mode, use protobuf for all other FEs
        if(FORCE_FRONTENDS_USE_PROTOBUF)
            set(OV_FRONTEND_PROTOBUF_LITE OFF)
        endif()
        # if protobuf::libprotobuf-lite is not available, use protobuf::libprotobuf
        if(NOT TARGET protobuf::libprotobuf-lite)
            set(OV_FRONTEND_PROTOBUF_LITE OFF)
        endif()

        if(OV_FRONTEND_PROTOBUF_LITE)
            set(protobuf_target_name libprotobuf-lite)
            set(protobuf_install_name "protobuf_lite_installed")
        else()
            set(protobuf_target_name libprotobuf)
            set(protobuf_install_name "protobuf_installed")
        endif()
        if(ENABLE_SYSTEM_PROTOBUF)
            # use imported target name with namespace
            set(protobuf_target_name "protobuf::${protobuf_target_name}")
        endif()

        ov_link_system_libraries(${TARGET_NAME} PRIVATE ${protobuf_target_name})

        # protobuf generated code emits -Wsuggest-override error
        if(SUGGEST_OVERRIDE_SUPPORTED)
            target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-suggest-override>)
        endif()

        # install protobuf if it is not installed yet
        if(NOT ${protobuf_install_name})
            if(ENABLE_SYSTEM_PROTOBUF)
                # we have to add find_package(Protobuf) to the OpenVINOConfig.cmake for static build
                # no needs to install protobuf
            else()
                ov_install_static_lib(${protobuf_target_name} ${OV_CPACK_COMP_CORE})
                set("${protobuf_install_name}" ON CACHE INTERNAL "" FORCE)
            endif()
        endif()
    endif()

    if(flatbuffers_schema_files)
        target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${flatbuffers_INCLUDE_DIRECTORIES})
    endif()

    ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME}
                               EXCLUDE_PATTERNS ${PROTO_SRCS} ${PROTO_HDRS} ${proto_files} ${flatbuffers_schema_files})

    # enable LTO
    set_target_properties(${TARGET_NAME} PROPERTIES
                          INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

    if(OV_FRONTEND_SKIP_NCC_STYLE)
        # frontend's CMakeLists.txt must define its own custom 'ov_ncc_naming_style' step
    else()
        ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                            SOURCE_DIRECTORIES "${frontend_root_dir}/include"
                                               "${frontend_root_dir}/src"
                            ADDITIONAL_INCLUDE_DIRECTORIES
                                $<TARGET_PROPERTY:${TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                $<TARGET_PROPERTY:${TARGET_NAME},INCLUDE_DIRECTORIES>)
    endif()

    add_dependencies(ov_frontends ${TARGET_NAME})

    # must be called after all target_link_libraries
    ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

    # since frontends are user-facing component which can be linked against,
    # then we need to mark it to be CXX ABI free
    ov_abi_free_target(${TARGET_NAME})

    # public target name
    set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::${OV_FRONTEND_NAME})

    # installation

    if(NOT OV_FRONTEND_SKIP_INSTALL)
        # convert option (OFF / ON) to actual value
        if(OV_FRONTEND_DISABLE_CPP_INSTALL)
            set(frontend_exclude_from_all EXCLUDE_FROM_ALL)
        endif()

        if(BUILD_SHARED_LIBS)
            # Note:
            # we use 'framework' as component for deployment scenario, i.e. for libraries itself
            # and use common 'core_dev' component for headers, cmake files and symlinks to versioned library
            set(lib_component "${OV_FRONTEND_NAME}")
            set(dev_component "${OV_CPACK_COMP_CORE_DEV}")

            ov_cpack_add_component(${lib_component} HIDDEN)

            if(OV_FRONTEND_LINKABLE_FRONTEND AND NOT OV_FRONTEND_DISABLE_CPP_INSTALL)
                set(export_set EXPORT OpenVINOTargets)
                set(archive_dest ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${dev_component} ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
                set(namelink NAMELINK_COMPONENT ${OV_CPACK_COMP_LINKS} ${OV_CPACK_COMP_LINKS_EXCLUDE_ALL})
            else()
                set(namelink NAMELINK_SKIP)
            endif()
            install(TARGETS ${TARGET_NAME} ${export_set}
                    RUNTIME DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${lib_component} ${frontend_exclude_from_all}
                    ${archive_dest}
                    LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR} COMPONENT ${lib_component} ${frontend_exclude_from_all}
                    ${namelink})

            # export to build tree
            # Note: we keep this even with passed DISABLE_CPP_INSTALL to ensure that Python API can be built
            if(OV_FRONTEND_LINKABLE_FRONTEND)
                export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
                       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
            endif()
        else()
            ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
        endif()

        if(OV_FRONTEND_LINKABLE_FRONTEND AND NOT OV_FRONTEND_DISABLE_CPP_INSTALL)
            # install library development files
            install(DIRECTORY ${${TARGET_NAME}_INCLUDE_DIR}/openvino
                    DESTINATION ${FRONTEND_INSTALL_INCLUDE}
                    COMPONENT ${dev_component}
                    ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL}
                    FILES_MATCHING PATTERN "*.hpp")
        endif()
    else()
        # skipped frontend has to be installed in static libraries case
        ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
    endif()
endmacro()
